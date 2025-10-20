#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查 Arrow 训练数据能否被你项目里的 tokenizer 正确处理，确保不存在 [UNK]。
- 读取：data/DrugLikeSMILSE-debug/train/data-00000-of-00001.arrow
- 词表：--vocab 指向你的 vocab.txt
- 分词器：src/smiles_bd/tokenizer_smiles.py 中的 RegexSmilesTokenizer

用法示例：
  python tools/check_no_unk_in_arrow.py \
    --arrow data/DrugLikeSMILSE-debug/train/data-00000-of-00001.arrow \
    --vocab /data/yqw/smiles-bd/vocab.txt \
    --src-root ./src \
    --column input \
    --max-report 50

退出码：
  0  -> 通过（无 [UNK]）
  1  -> 发现 [UNK] 或其他致命错误
"""

import os
import sys
import argparse
from typing import Optional, List, Tuple, Dict

def add_src_to_path(src_root: str):
    src_root = os.path.abspath(src_root)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

def load_tokenizer(vocab_path: str, src_root: str):
    add_src_to_path(src_root)
    try:
        from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
    except Exception as e:
        print("[ERROR] 无法从 src/smiles_bd/tokenizer_smiles.py 导入 RegexSmilesTokenizer：", e, file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(vocab_path):
        print(f"[ERROR] 词表文件不存在：{vocab_path}", file=sys.stderr)
        sys.exit(1)
    tok = RegexSmilesTokenizer(vocab_path)
    # 基础校验：必须包含核心符号
    required = ["[PAD]", "[MASK]", "[SEP]"]
    missing = [t for t in required if t not in tok.vocab]
    if missing:
        print(f"[ERROR] 词表缺少必要 token：{missing}。请确保包含 [PAD]/[MASK]/[SEP]。", file=sys.stderr)
        sys.exit(1)
    return tok

def open_arrow_reader(path: str):
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except Exception as e:
        print("[ERROR] 需要安装 pyarrow：pip install pyarrow；错误：", e, file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(path):
        print(f"[ERROR] Arrow 文件不存在：{path}", file=sys.stderr)
        sys.exit(1)

    # 优先按“文件模式”打开；失败则尝试“流模式”
    with open(path, "rb") as f:
        data = f.read(8)
    try:
        reader = ipc.open_file(path)  # RecordBatchFileReader
        mode = "file"
        return reader, mode
    except Exception:
        try:
            reader = ipc.open_stream(path)  # RecordBatchStreamReader
            mode = "stream"
            return reader, mode
        except Exception as e2:
            print("[ERROR] 既非 Arrow file 也非 stream：", e2, file=sys.stderr)
            sys.exit(1)

def infer_text_column_from_schema(schema, preferred: Optional[str] = None) -> str:
    # 优先使用用户指定
    if preferred is not None:
        if preferred in schema.names:
            return preferred
        else:
            raise KeyError(f"指定列 '{preferred}' 不存在。可用列：{list(schema.names)}")

    # 常见列名优先
    candidates = ["input", "text", "smiles", "SMILES", "sequence", "seq"]
    for c in candidates:
        if c in schema.names:
            return c

    # 否则挑第一个 string/large_string 列
    import pyarrow as pa
    for field in schema:
        if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
            return field.name

    raise KeyError(f"未能自动识别文本列，请使用 --column 指定。可用列：{list(schema.names)}")

def iter_batches(reader, mode: str):
    # 统一成 yield record_batch
    if mode == "file":
        for i in range(reader.num_record_batches):
            yield reader.get_record_batch(i)
    else:  # stream
        try:
            while True:
                rb = reader.read_next_batch()
                if rb is None:
                    break
                yield rb
        except StopIteration:
            return

def check_arrow_no_unk(arrow_path: str, vocab_path: str, src_root: str, column: Optional[str],
                       max_report: int = 20, max_show_tokens: int = 10) -> Tuple[int, Dict]:
    """
    返回 (bad_count, report_dict)
      - bad_count > 0 表示存在 [UNK]
      - report_dict 中包含详细统计与前若干个失败样例
    """
    tok = load_tokenizer(vocab_path, src_root)
    reader, mode = open_arrow_reader(arrow_path)

    # schema
    schema = reader.schema
    try:
        col_name = infer_text_column_from_schema(schema, column)
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)

    col_index = schema.get_field_index(col_name)
    total_rows = 0
    bad_rows = 0
    first_bad_examples = []  # (global_row_idx, sample_str, unknown_tokens[:K])

    # 遍历
    global_row_idx = 0
    for rb in iter_batches(reader, mode):
        arr = rb.column(col_index)
        # 转 Python 列表（避免一次性拷贝过大，可分块；但单 batch 通常可承受）
        py_list = arr.to_pylist()
        for s in py_list:
            total_rows += 1
            if s is None:
                # 跳过 None（若数据中允许空）
                global_row_idx += 1
                continue
            if not isinstance(s, str):
                s = str(s)

            # 直接用 tokenizer 的 regex 分词，逐 token 检查是否在 vocab
            tokens = tok.tokenize(s.replace(" ", ""))  # 与 encode 前处理一致
            unknown = [t for t in tokens if t not in tok.vocab]

            # 进一步稳妥：用 encode 看是否出现 unk_token_id
            ids = tok.encode(s, add_special_tokens=False, max_length=None, padding=False, truncation=False)
            has_unk_id = (tok.unk_token_id in ids) if hasattr(tok, "unk_token_id") else (len(unknown) > 0)

            if unknown or has_unk_id:
                bad_rows += 1
                if len(first_bad_examples) < max_report:
                    show_unk = unknown[:max_show_tokens]
                    first_bad_examples.append({
                        "row_index": global_row_idx,
                        "text_prefix": s[:200],
                        "unknown_tokens": show_unk,
                        "n_unknown": len(unknown),
                    })
            global_row_idx += 1

    report = {
        "arrow": arrow_path,
        "vocab": vocab_path,
        "column": col_name,
        "total_rows": total_rows,
        "bad_rows": bad_rows,
        "ok_rows": total_rows - bad_rows,
        "ratio_bad": (bad_rows / total_rows) if total_rows > 0 else 0.0,
        "examples": first_bad_examples,
        "required_tokens_in_vocab": {
            "PAD": "[PAD]" in tok.vocab,
            "MASK": "[MASK]" in tok.vocab,
            "SEP": "[SEP]" in tok.vocab,
        },
    }
    return bad_rows, report

def main():
    ap = argparse.ArgumentParser("检查 Arrow 训练数据经项目 tokenizer 编码后是否含 [UNK]")
    ap.add_argument("--arrow", type=str, required=True, help="Arrow 文件路径，如 data/.../data-00000-of-00001.arrow")
    ap.add_argument("--vocab", type=str, required=True, help="vocab.txt 路径")
    ap.add_argument("--src-root", type=str, default="./src", help="你的源码根目录（包含 smiles_bd/）")
    ap.add_argument("--column", type=str, default=None, help="文本列名（默认自动探测：优先 input/text/smiles/...，否则首个 string 列）")
    ap.add_argument("--max-report", type=int, default=20, help="最多展示多少条失败样例")
    ap.add_argument("--max-show-tokens", type=int, default=10, help="每条失败样例最多显示多少个未知 token")
    args = ap.parse_args()

    bad_count, report = check_arrow_no_unk(
        arrow_path=args.arrow,
        vocab_path=args.vocab,
        src_root=args.src_root,
        column=args.column,
        max_report=args.max_report,
        max_show_tokens=args.max_show_tokens,
    )

    # 打印摘要
    print("=" * 80)
    print("Arrow 文件：", report["arrow"])
    print("使用词表：  ", report["vocab"])
    print("文本列名：  ", report["column"])
    print(f"总行数：    {report['total_rows']}")
    print(f"异常行数：  {report['bad_rows']}  ({report['ratio_bad']*100:.4f}%)")
    print("词表关键符号存在性：", report["required_tokens_in_vocab"])
    print("=" * 80)

    if bad_count > 0:
        print("[FAIL] 发现含 [UNK] 的样本。前几例：")
        for ex in report["examples"]:
            print(f"- 行 {ex['row_index']}: 未知 token 数={ex['n_unknown']} | 前缀=\"{ex['text_prefix']}\"")
            print(f"  未知 token（最多显示 {args.max_show_tokens} 个）：{ex['unknown_tokens']}")
        sys.exit(1)
    else:
        print("[OK] 全部样本均可被 tokenizer 正确处理（无 [UNK]）。")
        sys.exit(0)

if __name__ == "__main__":
    main()
