import os, glob, torch
from torch.utils.data import Dataset, DataLoader

def _parquet_files(path: str):
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.parquet")))
    return sorted(glob.glob(path))  # 文件或通配符

class ParquetDataset(Dataset):
    """
    每行/条目应为已拼接好的统一长度的 SMILES 长序列（含 [EOS] 分隔与 [PAD] 到 max_len）
    """
    def __init__(self, path: str, tokenizer, max_len: int = 1024, text_col: str = None):
        import pandas as pd
        self.tok = tokenizer
        self.max_len = max_len

        files = _parquet_files(path)
        if not files: raise FileNotFoundError(f"No parquet files under: {path}")
        self.df = pd.concat((pd.read_parquet(p, engine="pyarrow") for p in files), ignore_index=True)

        if text_col is None:
            for c in ("text", "input"):
                if c in self.df.columns:
                    text_col = c
                    break
        self.col = text_col

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.at[int(idx), self.col]
        ids = self.tok.encode(text, add_special_tokens=False,max_length=self.max_len, padding=True, truncation=True)
        ids = torch.tensor(ids, dtype=torch.long)
        attn = (ids != self.tok.pad_token_id).long()
        return {"input_ids": ids, "attention_mask": attn}

def make_loaders(train_path: str, valid_path: str, tokenizer, max_len: int = 1024, batch_size: int = 2, num_workers: int = 0, text_col: str = None):
    train_ds = ParquetDataset(train_path, tokenizer, max_len=max_len, text_col=text_col)
    valid_ds = ParquetDataset(valid_path, tokenizer, max_len=max_len, text_col=text_col)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader