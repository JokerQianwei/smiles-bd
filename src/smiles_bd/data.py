
import os, json
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class TextLineDataset(Dataset):
    # Each line is a pre-concatenated sequence with [SEP] separators and [PAD] padded to max_len.
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

class ArrowDataset(Dataset):
    # Optional Arrow dataset: path should be a directory containing state.json with file list and column name.
    def __init__(self, path_dir: str, tokenizer, max_len: int = 1024, text_column: str = "input"):
        try:
            import pyarrow as pa, pyarrow.ipc as ipc
        except Exception as e:
            raise ImportError("pyarrow is required for ArrowDataset") from e
        self.tok = tokenizer
        self.max_len = max_len
        self.samples = []
        state_path = os.path.join(path_dir, "state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"state.json not found under {path_dir}")
        state = json.load(open(state_path, "r"))
        files = state.get("files", [])
        col = state.get("column", text_column)
        for f in files:
            with ipc.RecordBatchFileReader(open(os.path.join(path_dir, f), "rb")) as reader:
                for i in range(reader.num_record_batches):
                    rb = reader.get_batch(i).to_pydict()
                    texts = rb[col]
                    self.samples.extend(texts)
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

def _is_arrow_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "state.json"))

def make_loaders(train_path: str, valid_path: str, tokenizer, max_len: int = 1024, batch_size: int = 2,
                 num_workers: int = 0, pin_memory: bool = True, persistent_workers: bool = False,
                 prefetch_factor: int = 2, distributed: bool = False, seed: int = 42, text_column: str = "input"):
    if _is_arrow_dir(train_path) != _is_arrow_dir(valid_path):
        raise ValueError("Train/Valid must both be text or both be Arrow directories.")
    DS = ArrowDataset if _is_arrow_dir(train_path) else TextLineDataset
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
