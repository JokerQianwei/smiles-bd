
import argparse, os, math, time
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm 

from smiles_bd.config import load_config, merge_cli_overrides
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion
from smiles_bd.data import make_loaders
from smiles_bd.utils import (save_checkpoint, set_seed, set_torch_backends,
                             distributed_init, is_distributed, is_main_process, get_rank,
                             all_reduce_sum, sdpa_kernel_ctx, autocast_ctx, get_amp_dtype, count_parameters)

def parse_args():
    ap = argparse.ArgumentParser("Distributed training for SMILES masked diffusion (full-sequence SUBS).")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--override", type=str, nargs="*")
    return ap.parse_args()

def evaluate(diffuser, loader, device, amp_dtype):
    diffuser.eval()
    tot_nll = torch.zeros(1, device=device)
    tot_mask = torch.zeros(1, device=device)
    with torch.no_grad(), autocast_ctx(amp_dtype):
        if is_main_process():
            loader = tqdm(loader, desc="Evaluating")

        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            out = diffuser(batch)
            tot_nll += out.token_nll * out.num_masked
            tot_mask += out.num_masked.float()
    # reduce
    tot_nll = all_reduce_sum(tot_nll)
    tot_mask = all_reduce_sum(tot_mask)
    mean_nll = (tot_nll / torch.clamp_min(tot_mask, 1)).item()  # mean negative log-likelihood
    ppl = math.exp(mean_nll) if mean_nll < 30 else float("inf")  # masked-token CE
    diffuser.train()
    return ppl, mean_nll

def main():
    args = parse_args()
    cfg = merge_cli_overrides(load_config(args.config), args.override)

    dist_info = distributed_init()
    rank = get_rank()
    device = cfg["train"]["device"]
    if device == "cuda" and torch.cuda.is_available():
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    set_torch_backends()
    set_seed(42 + rank)

    tok = RegexSmilesTokenizer(cfg["paths"]["vocab_path"])

    train_loader, valid_loader, train_sampler, valid_sampler = make_loaders(
        cfg["paths"]["train_path"], cfg["paths"]["valid_path"], tok,
        max_len=cfg["model"]["max_len"], batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"], pin_memory=cfg["train"]["pin_memory"],
        persistent_workers=cfg["train"]["persistent_workers"], prefetch_factor=cfg["train"]["prefetch_factor"],
        distributed=is_distributed(), seed=42+rank, text_column=cfg["paths"].get("arrow_text_column","input")
    )

    model = TransformerDenoiser(
        vocab_size=tok.vocab_size, d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"],
        max_len=cfg["model"]["max_len"], dropout=cfg["model"]["dropout"],
        tie_embeddings=cfg["model"]["tie_embeddings"], disable_nested_tensor=cfg["model"]["disable_nested_tensor"]
    )
    schedule = ClippedLinearSchedule(beta=cfg["train"]["beta"], omega=cfg["train"]["omega"])
    diffuser = MaskedDiffusion(model, tok, schedule,
                               pad_token_id=tok.pad_token_id,
                               mask_token_id=tok.mask_token_id,
                               sep_token_id=tok.sep_token_id,
                               max_len=cfg["model"]["max_len"]).to(device)

    if is_main_process():
        total_params, trainable_params = count_parameters(diffuser)
        print(f"Total Parameters: {total_params / 1e6:.2f} M, Trainable Parameters: {trainable_params / 1e6:.2f} M")

    # optional compile
    if cfg["train"].get("compile", False) and hasattr(torch, "compile"):
        diffuser = torch.compile(diffuser, mode="reduce-overhead")

    # DDP
    if is_distributed():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        diffuser = DDP(diffuser, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Optimizer
    betas = tuple(cfg["train"].get("betas", (0.9, 0.95)))
    wd    = cfg["train"].get("weight_decay", 1.0e-2)
    eps   = cfg["train"].get("eps", 1.0e-8)
    optimizer = optim.AdamW(diffuser.parameters(), lr=float(cfg["train"]["lr"]), betas=betas, weight_decay=wd, eps=eps)

    # AMP dtype
    amp_dtype = get_amp_dtype(cfg["train"].get("amp", "auto"))
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    accum = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    steps_per_epoch = len(train_loader)

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        diffuser.train()

        if is_main_process():
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", total=steps_per_epoch)
        else:
            progress_bar = train_loader

        with sdpa_kernel_ctx(cfg["model"].get("sdpa_backend", "auto")):
            for step, batch in enumerate(progress_bar, 1):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with autocast_ctx(amp_dtype):
                    out = diffuser(batch)
                    loss = out.loss / accum
                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step % accum == 0:
                    if amp_dtype == torch.float16:
                        scaler.unscale_(optimizer)
                    clip_grad_norm_(diffuser.parameters(), cfg["train"]["grad_clip"])
                    if amp_dtype == torch.float16:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
                if is_main_process():
                    progress_bar.set_postfix(loss=out.loss.item(), lr=optimizer.param_groups[0]['lr'])

        ppl, valid_loss = evaluate(diffuser, valid_loader, device, amp_dtype)
        if is_main_process():
            print(f"[epoch {epoch}] valid loss={valid_loss:.4f}  valid ppl={ppl:.2f}")
            meta = {"vocab_size": tok.vocab_size, "max_len": cfg['model']['max_len'], **cfg["model"]}
            save_checkpoint(diffuser, os.path.join(cfg["paths"]["save_dir"], "model.pt"), meta=meta)

if __name__ == "__main__":
    main()
