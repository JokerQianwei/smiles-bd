import argparse, time, math, torch
from typing import List
from config import load_config, merge_cli_overrides
from tokenizer import RegexSmilesTokenizer
from model import TransformerDenoiser
from schedule import ClippedLinearSchedule
from diffusion import MaskedDiffusion
from utils import set_torch_backends, get_amp_dtype, autocast_ctx
from checkpoint import load_checkpoint

def parse_args():
    ap = argparse.ArgumentParser("Prefix-conditional sampling for SMILES clusters.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--override", type=str, nargs="*")
    ap.add_argument("--prefix", type=str, required=True)
    return ap.parse_args()


def strip_special_tokens(ids: List[int], tok: RegexSmilesTokenizer) -> List[int]:
    skip = {
        tok.pad_token_id,
        tok.bos_token_id,
        tok.eos_token_id,
    }
    return [int(t) for t in ids if int(t) not in skip]

def extract_first_sample_after_prefix(seq_ids: torch.Tensor, prefix_len: int, sep_id: int, pad_id: int) -> List[int]:
    # 取前缀之后的一段，遇到第一个 [SEP] 或 [PAD] 截断，返回 token id 列表（不含 SEP/PAD）
    arr = seq_ids.squeeze(0).tolist()
    tail = arr[prefix_len:]
    out = []
    for t in tail:
        if int(t) == sep_id or int(t) == pad_id:
            break
        out.append(int(t))
    return out

def main():
    args = parse_args()
    cfg = merge_cli_overrides(load_config(args.config), args.override)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 后端（进程级）：TF32 + SDPA
    set_torch_backends(cfg["train"].get("tf32", True), cfg["model"].get("sdpa_backend", "auto"))

    # AMP dtype（bf16/fp16/fp32)
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

    #  加载ckpt，只取模型权重
    payload = load_checkpoint(args.ckpt, map_location=device)
    getattr(diffuser, "module", diffuser).load_state_dict(payload["model"], strict=True)

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

    # 读取生成条数
    num_samples = int(cfg.get("sample", {}).get("num_samples", 1))

    collected_ids: List[List[int]] = []

    t_start = time.time()
    # 采样前向用 AMP（若 amp=bf16/fp16 会显著加速）
    with autocast_ctx(amp_dtype):
        while len(collected_ids) < num_samples:
            seq = diffuser.sample_with_prefix(
                prefix_ids,
                num_steps=cfg["sample"]["steps"],
                top_p=cfg["sample"]["top_p"]
            )
            ids = extract_first_sample_after_prefix(
                seq, prefix_len=prefix_ids.size(1), sep_id=tok.sep_token_id, pad_id=tok.pad_token_id
            )
            ids_clean = strip_special_tokens(ids, tok)
            if len(ids_clean) > 0:
                collected_ids.append(ids_clean)

    elapsed = time.time() - t_start

    # 去除特殊token（BOS/EOS/PAD），再decode
    samples = [tok.decode(ids) for ids in collected_ids]

    print("PREFIX:", args.prefix)
    print(f"SAMPLES (N={len(samples)}):")
    for i, c in enumerate(samples, 1):
        print(f"[{i}] {c}")
    print(f"Time:\t\t{elapsed:.2f} sec")

    # 使用 RDKit 判定有效性 + TDC 的 QED/SA 与 diversity
    import pandas as pd
    from tdc import Oracle, Evaluator
    from rdkit import Chem

    evaluator = Evaluator('diversity')
    oracle_qed = Oracle('qed')
    oracle_sa = Oracle('sa')

    # RDKit 有效性
    valid_flags = [Chem.MolFromSmiles(s) is not None for s in samples]
    valid_count = sum(1 for v in valid_flags if v)
    validity = valid_count / max(1, num_samples)

    # QED/SA 评估（对所有样本调用，但质量/多样性仅基于有效分子统计）
    qed_list = oracle_qed(samples)
    sa_list = oracle_sa(samples)

    df = pd.DataFrame({'smiles': samples, 'qed': qed_list, 'sa': sa_list, 'valid': valid_flags})
    df_valid = df[df['valid'] == True]
    df_unique = df_valid.drop_duplicates('smiles')
    uniqueness = (len(df_unique) / max(1, len(df_valid))) if len(df_valid) > 0 else 0.0
    if len(df_unique) >= 2:
        diversity = float(evaluator(df_unique['smiles'].tolist()))
        if not math.isfinite(diversity):
            diversity = 0.0
    else:
        diversity = 0.0
    df_quality = df_valid[(df_valid['qed'] >= 0.6) & (df_valid['sa'] <= 4)]
    quality = len(df_quality) / max(1, num_samples)

    print("[Evaluation by TDC]")
    print(f"Validity:\t{validity:.3f}")
    print(f"Uniqueness:\t{uniqueness:.3f}")
    print(f"Diversity:\t{diversity:.3f}")
    print(f"Quality:\t{quality:.3f}")

if __name__ == "__main__":
    main()
