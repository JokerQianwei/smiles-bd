
import math, torch, torch.nn as nn
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,L,D)

    def forward(self, x):  # (B,L,D)
        return x + self.pe[:, :x.size(1), :]

class TransformerDenoiser(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 8, max_len: int = 1024, dropout: float = 0.1,
                 tie_embeddings: bool = True, disable_nested_tensor: bool = True):
        super().__init__()
        self.max_len = max_len
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers,
            enable_nested_tensor=not disable_nested_tensor
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.lm_head.weight = self.emb.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        x = self.pos(self.emb(input_ids))
        kpm = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=kpm)
        return self.lm_head(x)
