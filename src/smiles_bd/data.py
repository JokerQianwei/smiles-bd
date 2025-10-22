
import os, json, glob
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class SmilesTokenizer(Dataset):
    # Optional Arrow dataset: path should be a directory containing state.json with file list.
    def __init__(self, path_dir: str, tokenizer, max_len: int = 1024, text_column: str = "input"):
        import pyarrow as pa, pyarrow.ipc as ipc; from tqdm import tqdm
        self.tok = tokenizer
        self.max_len = max_len
        self.col = text_column
        self.samples = []
        state_path = os.path.join(path_dir, "state.json")
        if not os.path.exists(state_path): raise FileNotFoundError(f"state.json not found under {path_dir}")
        state = json.load(open(state_path, "r"))
        files = [item["filename"] for item in state["_data_files"]]
        for f_name in tqdm(files, desc="Loading Arrow files"):
            file_path = os.path.join(path_dir, f_name)
            with open(file_path, "rb") as f:
                try:
                    with ipc.open_file(f) as reader:
                        for i in range(reader.num_record_batches):
                            batch = reader.get_batch(i)
                            texts = batch.column(self.col).to_pylist()
                            self.samples.extend(texts)
                except pa.ArrowInvalid:
                    f.seek(0) # 如果失败，将文件指针重置到开头
                    try:
                        with ipc.open_stream(f) as reader:
                            for batch in reader:
                                texts = batch.column(self.col).to_pylist()
                                self.samples.extend(texts)
                    except pa.ArrowInvalid as e_stream:
                        raise RuntimeError(f"File '{file_path}' is not a valid Arrow IPC File or Stream format.") from e_stream
                        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        ids = self.tok.encode(text, add_special_tokens=False, max_length=self.max_len, padding=True, truncation=True)
        ids = torch.tensor(ids, dtype=torch.long)
        attn = (ids != self.tok.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attn}

def make_loaders(train_path: str, valid_path: str, tokenizer, max_len: int = 1024, batch_size: int = 2,
                 num_workers: int = 0, pin_memory: bool = True, persistent_workers: bool = False,
                 prefetch_factor: int = 2, distributed: bool = False, seed: int = 42, text_column: str = "input"):

    train_ds = SmilesTokenizer(train_path, tokenizer, max_len=max_len)
    valid_ds = SmilesTokenizer(valid_path, tokenizer, max_len=max_len)

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
