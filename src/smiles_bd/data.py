
import torch
from torch.utils.data import Dataset, DataLoader

class TextLineDataset(Dataset):
    """
    Each line is an already concatenated SMILES sequence with [EOS] separators
    and padded with [PAD] to max_len. We do NOT inject BOS/EOS or wrap.
    """
    def __init__(self, path: str, tokenizer, max_len: int = 1024):
        self.path = path
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

def make_loaders(train_path: str, valid_path: str, tokenizer, max_len: int = 1024, batch_size: int = 2, num_workers: int = 0):
    train_ds = TextLineDataset(train_path, tokenizer, max_len=max_len)
    valid_ds = TextLineDataset(valid_path, tokenizer, max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, valid_loader
