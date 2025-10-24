import argparse, torch
from config import load_config, merge_cli_overrides
from tokenizer import RegexSmilesTokenizer
from model import TransformerDenoiser
from schedule import ClippedLinearSchedule
from diffusion import MaskedDiffusion
from utils import load_checkpoint, set_torch_backends, get_amp_dtype, autocast_ctx

def parse_args():
    ap = argparse.ArgumentParser("Prefix-conditional sampling for SMILES clusters.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--override", type=str, nargs="*")
    ap.add_argument("--prefix", type=str, required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = merge_cli_overrides(load_config(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 后端（进程级）：TF32 + SDPA
    set_torch_backends(cfg["train"].get("tf32", True), cfg["model"].get("sdpa_backend", "auto"))

    # AMP dtype（bf16/fp16/fp32）----
    amp_dtype = get_amp_dtype(cfg["train"].get("amp", "auto"))

    tok = RegexSmilesTokenizer(cfg["paths"]["vocab_path"])
    model = TransformerDenoiser(
        vocab_size=tok.vocab_size, d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"],
        max_len=cfg["model"]["max_len"], dropout=cfg["model"]["dropout"],
        tie_embeddings=cfg["model"]["tie_embeddings"]
    )
    schedule = ClippedLinearSchedule(beta=cfg["train"]["beta"], omega=cfg["train"]["omega"])
    diffuser = MaskedDiffusion(
        model, tok, schedule,
        pad_token_id=tok.pad_token_id,
        mask_token_id=tok.mask_token_id,
        sep_token_id=tok.sep_token_id,
        max_len=cfg["model"]["max_len"]
    ).to(device)

    load_checkpoint(diffuser, args.ckpt, map_location=device)
    diffuser.eval()

    prefix_ids = torch.tensor(
        tok.encode(
            args.prefix,
            insert_special_tokens=False,
            max_length=cfg["model"]["max_len"],
            padding=False,
            truncation=True
        ),
        dtype=torch.long, device=device
    ).unsqueeze(0)

    # 采样前向用 AMP（若 amp=bf16/fp16 会显著加速）
    with autocast_ctx(amp_dtype):
        seq = diffuser.sample_with_prefix(
            prefix_ids,
            num_steps=cfg["sample"]["steps"],
            top_p=cfg["sample"]["top_p"]
        )

    candidates_ids = diffuser.split_candidates_after_prefix(seq, prefix_len=prefix_ids.size(1))
    candidates = [tok.decode(ids) for ids in candidates_ids]

    print("PREFIX:", args.prefix)
    print("CANDIDATES:")
    for i, c in enumerate(candidates, 1):
        print(f"[{i}] {c}")

if __name__ == "__main__":
    main()
