
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,L,D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerDenoiser(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 8, max_len: int = 1024, dropout: float = 0.1, tie_weights: bool = True):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=4*d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        x = self.token_emb(input_ids)
        x = self.pos(x)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.lm_head(x)
