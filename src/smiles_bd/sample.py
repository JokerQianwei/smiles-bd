
import argparse, torch
from smiles_bd.config import load_config, merge_cli_overrides
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion
from smiles_bd.utils import load_checkpoint, peek_meta

def parse_args():
    ap = argparse.ArgumentParser("Prefix-conditional sampling for SMILES clusters.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to saved diffuser checkpoint (.pt).")
    ap.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config (for vocab path & defaults).")
    ap.add_argument("--override", type=str, nargs="*", help="Optional dotted overrides, e.g. sample.steps=32")
    ap.add_argument("--prefix", type=str, required=True, help="SMILES A; placed at start and frozen.")
    return ap.parse_args()

def _parse_overrides(arg_list):
    if not arg_list:
        return {}
    out = {}
    for item in arg_list:
        if "=" not in item: continue
        k, v = item.split("=", 1)
        out[k] = v
    return out

def main():
    args = parse_args()
    cfg = merge_cli_overrides(load_config(args.config), _parse_overrides(args.override))

    meta = peek_meta(args.ckpt, map_location="cpu")
    tok = RegexSmilesTokenizer(cfg["paths"]["vocab_path"])

    # Prefer checkpoint meta
    model_kwargs = dict(d_model=cfg["model"]["d_model"],
                        n_heads=cfg["model"]["n_heads"],
                        n_layers=cfg["model"]["n_layers"],
                        max_len=cfg["model"]["max_len"],
                        dropout=cfg["model"]["dropout"])
    if meta:
        for k in ["d_model","n_heads","n_layers","max_len","dropout"]:
            if k in meta:
                model_kwargs[k] = meta[k]

    model = TransformerDenoiser(vocab_size=tok.vocab_size, **model_kwargs)
    schedule = ClippedLinearSchedule()
    diffuser = MaskedDiffusion(model, tok, schedule,
                               pad_token_id=tok.pad_token_id,
                               mask_token_id=tok.mask_token_id,
                               eos_token_id=tok.eos_token_id,
                               max_len=model_kwargs["max_len"])

    device = cfg["sample"]["device"]
    diffuser.to(device).eval()
    load_checkpoint(diffuser, args.ckpt, map_location=device)

    prefix_ids = torch.tensor(tok.encode(args.prefix, add_special_tokens=False, max_length=model_kwargs["max_len"], padding=False, truncation=True), dtype=torch.long, device=device).unsqueeze(0)
    seq = diffuser.sample_with_prefix(prefix_ids, num_steps=cfg["sample"]["steps"], top_p=cfg["sample"]["top_p"])
    candidates_ids = diffuser.split_candidates_after_prefix(seq, prefix_len=prefix_ids.size(1))
    candidates = [tok.decode(ids) for ids in candidates_ids]
    print("PREFIX:", args.prefix)
    print("CANDIDATES:")
    for i, c in enumerate(candidates, 1):
        print(f"[{i}] {c}")

if __name__ == "__main__":
    main()
