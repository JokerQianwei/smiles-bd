import argparse, os, torch, torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from smiles_bd.config import load_config, merge_cli_overrides
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion
from smiles_bd.data import make_loaders
from smiles_bd.utils import save_checkpoint, set_seed

def parse_args():
    ap = argparse.ArgumentParser("Train masked diffusion for SMILES clusters (full-sequence).")
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--override", type=str, nargs="*")
    return ap.parse_args()

@torch.no_grad()
def evaluate(diffuser, valid_loader, device):
    diffuser.eval()
    tot_loss, tot_mask = 0.0, 0
    for batch in valid_loader:
        batch = {k:v.to(device) for k,v in batch.items()}
        out = diffuser.training_step(batch)
        tot_loss += out.token_nll.item() * int(out.num_masked)
        tot_mask += int(out.num_masked)
    diffuser.train()
    if tot_mask == 0: return float("inf")
    nll = tot_loss / tot_mask
    ppl = torch.exp(torch.tensor(nll)).item()
    return ppl

def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, args.override)

    set_seed(42)
    device = cfg["train"]["device"]
    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)

    tok = RegexSmilesTokenizer(cfg["paths"]["vocab_path"])
    train_loader, valid_loader = make_loaders(
        cfg["paths"]["train_path"], cfg["paths"]["valid_path"], tok,
        max_len=cfg["model"]["max_len"], batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"]
    )

    model = TransformerDenoiser(
        vocab_size=tok.vocab_size, d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"],
        max_len=cfg["model"]["max_len"], dropout=cfg["model"]["dropout"],
        tie_weights=cfg["model"].get("tie_weights", True)
    ).to(device)

    schedule = ClippedLinearSchedule(beta=cfg["train"]["beta"], omega=cfg["train"]["omega"])
    diffuser = MaskedDiffusion(model, tok, schedule,
        pad_token_id=tok.pad_token_id, mask_token_id=tok.mask_token_id, sep_token_id=tok.sep_token_id,
        max_len=cfg["model"]["max_len"]).to(device)

    optimizer = optim.AdamW(diffuser.parameters(), lr=cfg["train"]["lr"])
    for epoch in range(1, cfg["train"]["epochs"]+1):
        for step, batch in enumerate(train_loader, 1):
            batch = {k:v.to(device) for k,v in batch.items()}
            out = diffuser.training_step(batch)
            out.loss.backward()
            clip_grad_norm_(diffuser.parameters(), cfg["train"]["grad_clip"])
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if step % 50 == 0:
                print(f"[epoch {epoch}] step {step} loss={out.loss.item():.4f} masked={int(out.num_masked)}")
        ppl = evaluate(diffuser, valid_loader, device)
        print(f"[epoch {epoch}] valid ppl (masked-token CE) ~ {ppl:.2f}")
        meta = {"vocab_size": tok.vocab_size, "max_len": cfg["model"]["max_len"], **cfg["model"]}
        save_checkpoint(diffuser, os.path.join(cfg["paths"]["save_dir"], "model.pt"), meta=meta)

if __name__ == "__main__":
    main()
