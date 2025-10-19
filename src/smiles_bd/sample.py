import argparse, torch
from smiles_bd.config import load_config, merge_cli_overrides
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion
from smiles_bd.utils import load_checkpoint

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
    device = cfg["sample"]["device"]

    tok = RegexSmilesTokenizer(cfg["paths"]["vocab_path"])
    model = TransformerDenoiser(
        vocab_size=tok.vocab_size, d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"], n_layers=cfg["model"]["n_layers"],
        max_len=cfg["model"]["max_len"], dropout=cfg["model"]["dropout"],
        tie_weights=cfg["model"].get("tie_weights", True)
    )
    schedule = ClippedLinearSchedule(beta=cfg["train"]["beta"], omega=cfg["train"]["omega"])
    diffuser = MaskedDiffusion(model, tok, schedule,
        pad_token_id=tok.pad_token_id, mask_token_id=tok.mask_token_id, eos_token_id=tok.eos_token_id,
        max_len=cfg["model"]["max_len"])
    diffuser.to(device).eval()
    load_checkpoint(diffuser, args.ckpt, map_location=device)

    # 前缀 A：建议自行在末尾拼上 [EOS]，或使用纯 A（看你的约定）
    prefix_txt = args.prefix
    prefix_ids = torch.tensor(tok.encode(prefix_txt, add_special_tokens=False,
                          max_length=cfg["model"]["max_len"], padding=False, truncation=True),
                          dtype=torch.long, device=device).unsqueeze(0)

    seq = diffuser.sample_with_prefix(prefix_ids, num_steps=cfg["sample"]["steps"], top_p=cfg["sample"]["top_p"])
    candidates_ids = diffuser.split_candidates_after_prefix(seq, prefix_len=prefix_ids.size(1))
    candidates = [tok.decode(ids) for ids in candidates_ids]
    print("PREFIX:", args.prefix)
    print("CANDIDATES:")
    for i, c in enumerate(candidates, 1):
        print(f"[{i}] {c}")

if __name__ == "__main__":
    main()