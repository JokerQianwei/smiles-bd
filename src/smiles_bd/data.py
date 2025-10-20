import os, json, glob, torch
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def _arrow_files(path: str):
    """收集 .arrow 文件列表；支持目录或单文件。"""
    if os.path.isdir(path):
        state_path = os.path.join(path, "state.json")
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            return [os.path.join(path, x["filename"]) for x in state.get("_data_files", [])]
        return sorted(glob.glob(os.path.join(path, "*.arrow")))
    return [path] if path.endswith(".arrow") else []

class ArrowDataset(Dataset):
    """从 HuggingFace Arrow（.arrow + state.json）读取 `input` 列并分词。"""
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.tok = tokenizer
        self.max_len = max_len
        self.col = "input"

        files = _arrow_files(path)
        if not files: raise FileNotFoundError(f"未找到 .arrow 文件: {path}")

        tables = []
        for p in files:
            mm = pa.memory_map(p, "r")
            try:
                reader = pa_ipc.RecordBatchFileReader(mm)
            except pa.lib.ArrowInvalid:
                reader = pa_ipc.open_stream(mm)
            tables.append(reader.read_all())
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        self.df = table.to_pandas()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.at[int(idx), self.col]
        ids = self.tok.encode(text, add_special_tokens=False, max_length=self.max_len, padding=True, truncation=True,)
        ids = torch.tensor(ids, dtype=torch.long)
        attn = (ids != self.tok.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attn}

def make_loaders(train_path: str, valid_path: str, tokenizer, max_len: int = 1024, batch_size: int = 2, num_workers: int = 0,):
    train_ds = ArrowDataset(train_path, tokenizer, max_len=max_len)
    valid_ds = ArrowDataset(valid_path, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader
