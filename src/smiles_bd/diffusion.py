
from dataclasses import dataclass
from typing import List, Tuple
import torch, torch.nn.functional as F

@dataclass
class TrainOutput:
    loss: torch.Tensor
    token_nll: torch.Tensor
    num_masked: torch.Tensor

class MaskedDiffusion(torch.nn.Module):
    def __init__(self, model, tokenizer, schedule, pad_token_id: int, mask_token_id: int, sep_token_id: int, max_len: int = 1024):
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self.schedule = schedule
        self.pad_id = pad_token_id
        self.mask_id = mask_token_id
        self.sep_id = sep_token_id
        self.max_len = max_len

    def forward(self, batch) -> TrainOutput:
        return self.training_step(batch)

    def _mask_with_rate(self, x0: torch.Tensor, attn: torch.Tensor, mask_rate: torch.Tensor):
        B, L = x0.shape
        probs = mask_rate.view(B, 1).expand(B, L)
        bern = torch.bernoulli(probs).to(x0.dtype)
        mask_positions = (bern == 1) & (attn == 1)
        xt = x0.clone()
        xt[mask_positions] = self.mask_id
        return xt, mask_positions

    def training_step(self, batch) -> TrainOutput:
        x0 = batch["input_ids"]
        attn = batch["attention_mask"]
        B, L = x0.shape
        device = x0.device
        mask_rate = self.schedule.sample_mask_rate((B,), device=device)
        xt, mask_positions = self._mask_with_rate(x0, attn, mask_rate)
        logits = self.model(xt, attention_mask=attn)
        # forbid emitting PAD/MASK tokens
        logits[..., self.pad_id]  = -1e9
        logits[..., self.mask_id] = -1e9
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x0.view(-1), reduction="none").view(B, L)
        masked_loss = (ce * mask_positions.float())
        per_ex = masked_loss.sum(dim=1) / mask_positions.sum(dim=1).clamp_min(1)
        w = self.schedule.loss_weight(mask_rate)
        loss = (per_ex * w).mean()
        token_nll = masked_loss.sum() / mask_positions.sum().clamp_min(1)
        return TrainOutput(loss=loss, token_nll=token_nll.detach(), num_masked=mask_positions.sum())

    @torch.no_grad()
    def _nucleus_pick(self, probs_row: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
        # probs_row: (K,V)
        sp, si = torch.sort(probs_row, dim=-1, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        sp = sp * keep
        denom = sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        sp = sp / denom
        idx = torch.multinomial(sp, 1).squeeze(-1)
        return si.gather(1, idx.unsqueeze(-1)).squeeze(-1)

    @torch.no_grad()
    def sample_with_prefix(self, prefix_ids: torch.Tensor, num_steps: int = 24, top_p: float = 0.9) -> torch.Tensor:
        if prefix_ids.ndim == 1: prefix_ids = prefix_ids.unsqueeze(0)
        B, pref_len = prefix_ids.size()
        L = self.max_len
        device = next(self.parameters()).device
        x = torch.full((B, L), self.mask_id, dtype=torch.long, device=device)
        x[:, :pref_len] = prefix_ids
        attn = (x != self.pad_id).long()
        revealed = torch.zeros(B, L, dtype=torch.bool, device=device); revealed[:, :pref_len] = True

        for s in range(num_steps):
            logits = self.model(x, attention_mask=attn)
            probs = torch.softmax(logits, dim=-1)
            probs[..., self.pad_id]  = 0
            probs[..., self.mask_id] = 0

            masked_positions = (x == self.mask_id)
            if masked_positions.sum() == 0: break

            remain = masked_positions.sum(dim=1)
            k = (remain.float() / max(1, (num_steps - s))).ceil().clamp_min(1).long()

            max_prob, _ = probs.max(dim=-1)
            for b in range(B):
                if remain[b] == 0: continue
                cand_idx = torch.where(masked_positions[b])[0]
                cand_idx = cand_idx[cand_idx >= pref_len]  # freeze prefix
                if cand_idx.numel() == 0: continue
                conf = max_prob[b, cand_idx]
                topk = min(k[b].item(), cand_idx.numel())
                chosen = cand_idx[torch.topk(conf, topk).indices]
                picked = self._nucleus_pick(probs[b, chosen], top_p=top_p)
                x[b, chosen] = picked
                revealed[b, chosen] = True
        return x

    @torch.no_grad()
    def split_candidates_after_prefix(self, seq_ids: torch.Tensor, prefix_len: int) -> List[List[int]]:
        gen = seq_ids.squeeze(0).tolist()
        tail = gen[prefix_len:]
        out, cur = [], []
        for t in tail:
            if t == self.sep_id:
                if cur: out.append(cur); cur = []
            elif t == self.pad_id:
                continue
            else:
                cur.append(t)
        if cur: out.append(cur)
        return out
