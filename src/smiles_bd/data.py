
import os, json, glob
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class TextLineDataset(Dataset):
    # 为 pytest 保留
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len
        with open(path, "r", encoding="utf-8") as f:
            self.lines = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        ids = self.tok.encode(text, add_special_tokens=False, max_length=self.max_len, padding=True, truncation=True)
        ids = torch.tensor(ids, dtype=torch.long)
        attn = (ids != self.tok.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attn}

def _arrow_files(path: str):
    """收集 .arrow 文件列表；支持目录或单文件。"""
    if os.path.isdir(path):
        state_path = os.path.join(path, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            files = [os.path.join(path, x["filename"]) for x in state.get("_data_files", [])]
            if files:
                return files
        return sorted(glob.glob(os.path.join(path, "*.arrow")))
    return [path] if path.endswith(".arrow") else []

class ArrowDataset(Dataset):
    """从 HuggingFace Arrow（.arrow + state.json）读取指定列并分词。"""
    def __init__(self, path: str, tokenizer, max_len: int = 1024, text_column: str = "input"):
        import pyarrow as pa, pyarrow.ipc as ipc
        self.tok = tokenizer
        self.max_len = max_len
        self.col = text_column

        files = _arrow_files(path)
        if not files:
            raise FileNotFoundError(f"未找到 .arrow 文件: {path}")

        tables = []
        for p in files:
            mm = pa.memory_map(p, "r")
            try:
                reader = ipc.RecordBatchFileReader(mm)
            except pa.lib.ArrowInvalid:
                reader = ipc.open_stream(mm)
            tables.append(reader.read_all())
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        # 直接取列为 Python list，避免额外依赖
        self.samples = table[self.col].to_pylist()
        if not self.samples:
            raise RuntimeError("ArrowDataset loaded zero samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tok.encode(text, add_special_tokens=False, max_length=self.max_len, padding=True, truncation=True)
        ids = torch.tensor(ids, dtype=torch.long)
        attn = (ids != self.tok.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attn}

def _has_arrow_files(path: str) -> bool:
    return bool(_arrow_files(path))

def make_loaders(train_path: str, valid_path: str, tokenizer, max_len: int = 1024, batch_size: int = 2,
                 num_workers: int = 0, pin_memory: bool = True, persistent_workers: bool = False,
                 prefetch_factor: int = 2, distributed: bool = False, seed: int = 42, text_column: str = "input"):
    if _has_arrow_files(train_path) != _has_arrow_files(valid_path):
        raise ValueError("Train/Valid must both be text or both be Arrow directories.")
    DS = ArrowDataset if _has_arrow_files(train_path) else TextLineDataset
    train_ds = DS(train_path, tokenizer, max_len=max_len) if DS is TextLineDataset else DS(train_path, tokenizer, max_len=max_len, text_column=text_column)
    valid_ds = DS(valid_path, tokenizer, max_len=max_len) if DS is TextLineDataset else DS(valid_path, tokenizer, max_len=max_len, text_column=text_column)

    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False, seed=seed)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False, drop_last=False, seed=seed)
    else:
        train_sampler = valid_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        sampler=train_sampler, prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        sampler=valid_sampler, prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    return train_loader, valid_loader, train_sampler, valid_sampler
