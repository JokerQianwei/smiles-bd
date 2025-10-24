import os
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from datasets import load_dataset, load_from_disk, DatasetDict
import torch
from torch.utils.data import DataLoader, DistributedSampler

@dataclass
class DataPaths:
    raw_data_dir: str
    cache_dir: str

def _discover_raw_txt(raw_dir: str) -> Tuple[Optional[str], Optional[str]]:
    train_txt = os.path.join(raw_dir, "train.txt")
    valid_txt = os.path.join(raw_dir, "valid.txt")
    if os.path.isfile(train_txt) and os.path.isfile(valid_txt):
        return train_txt, valid_txt
    return None, None

def _looks_like_hf_disk(path: str) -> bool:
    if not os.path.isdir(path): return False
    # single dataset
    if any(os.path.exists(os.path.join(path, f)) for f in ["dataset_info.json", "state.json"]): return True
    # dataset dict (train/, validation/ ...)
    for sub in os.listdir(path):
        sub_path = os.path.join(path, sub)
        if os.path.isdir(sub_path) and any(os.path.exists(os.path.join(sub_path, f)) for f in ["dataset_info.json", "state.json"]): return True
    return False

def prepare_or_load_dataset(raw_data_dir: str, cache_dir: str, tokenizer, max_len: int,
                            text_column: str = "text", num_proc: int = len(os.sched_getaffinity(0)),
                            insert_special_tokens: bool = False) -> DatasetDict:
    """
    If `cache_dir` has a saved HF DatasetDict, load and return.
    Else, build from raw_data_dir (train.txt/valid.txt or from pre-saved HF dataset),
    tokenize with datasets.map (batched + multiprocessing), save to `cache_dir`, and return.
    """
    os.makedirs(cache_dir, exist_ok=True)
    # 1) Load from cache if present
    if os.path.isdir(cache_dir) and _looks_like_hf_disk(cache_dir):
        return load_from_disk(cache_dir)

    # 2) Build from raw
    tr_txt, va_txt = _discover_raw_txt(raw_data_dir)
    if tr_txt and va_txt:
        dset = load_dataset("text", data_files={"train": tr_txt, "validation": va_txt})
    else:
        # raw might itself be HF saved dataset dict (with train/validation subdirs) or a single dataset
        if _looks_like_hf_disk(raw_data_dir):
            dset = load_from_disk(raw_data_dir)
            if not isinstance(dset, DatasetDict):
                dset = DatasetDict({"train": dset, "validation": dset})
        else:
            # allow nested subdirs: raw/train, raw/validation
            tr_dir = os.path.join(raw_data_dir, "train")
            va_dir = os.path.join(raw_data_dir, "validation")
            if _looks_like_hf_disk(tr_dir) and _looks_like_hf_disk(va_dir):
                from datasets import Dataset
                dset = DatasetDict({"train": load_from_disk(tr_dir), "validation": load_from_disk(va_dir)})
            else:
                raise FileNotFoundError(f"Could not find train/valid data under {raw_data_dir}")

    # 3) Tokenize with map (batched)
    def _map_fn(batch):
        return tokenizer(batch, text_key=text_column, max_length=max_len, padding=True, truncation=True, insert_special_tokens=insert_special_tokens)

    cols_to_remove = [c for c in dset["train"].column_names if c != text_column]
    tokenized = dset.map(_map_fn, batched=True, num_proc=num_proc, remove_columns=cols_to_remove)

    # 4) Set torch format for fast DataLoader
    tokenized = tokenized.with_format(type="torch", columns=["input_ids", "attention_mask"])

    # 5) Save to cache_dir
    tokenized.save_to_disk(cache_dir)
    return tokenized

def create_dataloaders(tokenized: DatasetDict, batch_size: int, num_workers: int = 4,
                       pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler], Optional[DistributedSampler]]:
    train_ds = tokenized["train"]
    valid_ds = tokenized["validation"] if "validation" in tokenized else tokenized["valid"]

    # DDP-aware samplers
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        valid_sampler = DistributedSampler(valid_ds, shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
                              sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                              sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, train_sampler, valid_sampler
