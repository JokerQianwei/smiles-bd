import argparse, os, math, json
from typing import Dict, Any
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .config import load_config, merge_cli_overrides
from .tokenizer import RegexSmilesTokenizer
from .model import TransformerDenoiser
from .schedule import ClippedLinearSchedule
from .diffusion import MaskedDiffusion
from .data import prepare_or_load_dataset, create_dataloaders
from .checkpoint import pack_state, save_checkpoint, load_checkpoint
from .utils import (set_seed, set_torch_backends, distributed_init, is_distributed, is_main_process,
                    get_amp_dtype, autocast_ctx, all_reduce_sum, count_parameters, print0)

@torch.no_grad()
def evaluate(diffuser, loader, device, amp_dtype):
    tot_nll = torch.zeros(1, device=device)
    tot_mask = torch.zeros(1, device=device)
    if is_main_process():
        loader = tqdm(loader, desc="valid")
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast_ctx(amp_dtype):
            out = diffuser(batch)
        tot_nll += out.token_nll * out.num_masked
        tot_mask += out.num_masked.float()
    tot_nll = all_reduce_sum(tot_nll)
    tot_mask = all_reduce_sum(tot_mask)
    mean_nll = (tot_nll / torch.clamp_min(tot_mask, 1)).item()
    ppl = math.exp(mean_nll) if mean_nll < 30 else float("inf")
    return ppl, mean_nll

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="override paths.data_dir from config")
    parser.add_argument("--cache_dir", type=str, default=None, help="override paths.cache_dir from config")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("--overrides", type=str, default="{}", help="JSON string to override config keys")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, json.loads(args.overrides))

    if args.data_dir:
        cfg.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.cache_dir:
        cfg.setdefault("paths", {})["cache_dir"] = args.cache_dir

    distributed_init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["train"].get("seed", 1337))
    set_torch_backends()

    # Tokenizer + datasets
    tok = RegexSmilesTokenizer(cfg["paths"]["vocab_path"])
    tokenized = prepare_or_load_dataset(raw_data_dir=cfg["paths"]["data_dir"],
                                        cache_dir=cfg["paths"]["cache_dir"],
                                        tokenizer=tok,
                                        max_len=cfg["model"]["max_len"],
                                        text_column=cfg["data"].get("text_column", "text"),
                                        num_proc=cfg["data"].get("num_proc", len(os.sched_getaffinity(0))))
    train_loader, valid_loader, train_sampler, valid_sampler = create_dataloaders(
        tokenized, batch_size=cfg["train"]["batch_size"], num_workers=cfg["data"].get("num_workers", 8)
    )

    # Model + diffusion
    model = TransformerDenoiser(
        vocab_size=tok.vocab_size, max_len=cfg["model"]["max_len"],
        d_model=cfg["model"]["d_model"], n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"], dropout=cfg["model"].get("dropout", 0.1)
    ).to(device)
    schedule = ClippedLinearSchedule(beta=cfg["train"]["beta"], omega=cfg["train"]["omega"])
    diffuser = MaskedDiffusion(model, tok, schedule,
                               pad_token_id=tok.pad_token_id,
                               mask_token_id=tok.mask_token_id,
                               sep_token_id=tok.sep_token_id,
                               max_len=cfg["model"]["max_len"]).to(device)

    if is_main_process():
        total_params, trainable_params = count_parameters(diffuser)
        print0(f"Params: total {total_params/1e6:.2f}M, trainable {trainable_params/1e6:.2f}M")

    # Compile for speed (optional)
    if cfg["train"].get("compile", False) and hasattr(torch, "compile"):
        diffuser = torch.compile(diffuser, mode=cfg["train"].get("compile_mode", "reduce-overhead"))

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

    # Resume
    start_step = 0
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, map_location="cpu")
        sd = ckpt.get("model", ckpt.get("state_dict", None))
        if sd is not None:
            getattr(diffuser, "module", diffuser).load_state_dict(sd)
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        print0(f"Resumed from {args.resume} at step={start_step}, best_val={best_val:.4f}")

    # Training schedule
    max_iters      = int(cfg["train"]["max_iters"])
    grad_accum     = int(cfg["train"].get("grad_accum_steps", 1))
    log_interval   = int(cfg["train"].get("log_interval", 10))
    eval_interval  = int(cfg["train"].get("eval_interval", 1000))
    save_interval  = int(cfg["train"].get("save_interval", 1000))
    save_dir       = cfg["paths"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    if is_distributed() and train_sampler is not None:
        train_sampler.set_epoch(0)

    iter_loader = infinite_loader(train_loader)

    step = start_step
    diffuser.train()
    progress = tqdm(total=max_iters - step, initial=0, disable=not is_main_process(), desc="train")
    while step < max_iters:
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            if is_distributed() and train_sampler is not None and (step % 1000 == 0):
                train_sampler.set_epoch(step)
            batch = next(iter_loader)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast_ctx(amp_dtype):
                out = diffuser(batch)
                loss = out.loss / grad_accum
            if amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if amp_dtype == torch.float16:
            scaler.unscale_(optimizer)
        if cfg["train"].get("grad_clip", 0.0) > 0:
            clip_grad_norm_(diffuser.parameters(), cfg["train"]["grad_clip"])
        if amp_dtype == torch.float16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        step += 1
        if is_main_process() and step % log_interval == 0:
            progress.update(log_interval)
            progress.set_postfix(step=step, loss=float(out.loss))

        # periodic eval
        if step % eval_interval == 0 or step == max_iters:
            diffuser.eval()
            ppl, val_loss = evaluate(diffuser, valid_loader, device, amp_dtype)
            if is_main_process():
                print0(f"[iter {step}] valid loss={val_loss:.4f}, ppl={ppl:.2f}")
            diffuser.train()
            # save best
            if val_loss < best_val and is_main_process():
                best_val = val_loss
                state = pack_state(diffuser, optimizer, scaler, cfg, step, best_val)
                save_checkpoint(os.path.join(save_dir, "best_model.pt"), state)

        # periodic save
        if is_main_process() and (step % save_interval == 0 or step == max_iters):
            state = pack_state(diffuser, optimizer, scaler, cfg, step, best_val)
            save_checkpoint(os.path.join(save_dir, f"iter_{step:07d}.pt"), state)

    if is_main_process():
        progress.close()
        print0("Training done.")

if __name__ == "__main__":
    main()
