from dataclasses import dataclass
from typing import List
import torch, torch.nn.functional as F

@dataclass
class TrainOutput:
    loss: torch.Tensor
    token_nll: torch.Tensor
    num_masked: torch.Tensor

class MaskedDiffusion(torch.nn.Module):
    def __init__(self, model, tokenizer, schedule, pad_token_id, mask_token_id, eos_token_id, max_len=1024):
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self.schedule = schedule
        self.pad_id, self.mask_id, self.eos_id = pad_token_id, mask_token_id, eos_token_id
        self.max_len = max_len

    def training_step(self, batch):
        x0 = batch["input_ids"].to(next(self.model.parameters()).device)  # (B,L)
        attn = batch["attention_mask"].to(x0.device)                     # (B,L)
        B,L = x0.shape
        r = self.schedule.sample_mask_rate((B,), x0.device)              # (B,)
        mask_pos = (torch.rand(B,L,device=x0.device) < r.view(B,1)) & (attn==1)
        x_in = x0.clone(); x_in[mask_pos] = self.mask_id

        logits = self.model(x_in, attention_mask=attn)                   # (B,L,V)
        logits[:, :, self.pad_id]  = -1e9
        logits[:, :, self.mask_id] = -1e9

        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x0.view(-1), reduction="none").view(B,L)
        masked_loss = (ce * mask_pos.float())
        per_ex = masked_loss.sum(dim=1) / mask_pos.sum(dim=1).clamp_min(1)
        w = self.schedule.loss_weight(r)                                  # (B,)
        loss = (per_ex * w).mean()
        token_nll = masked_loss.sum() / mask_pos.sum().clamp_min(1)
        return TrainOutput(loss=loss, token_nll=token_nll, num_masked=mask_pos.sum())

    @torch.no_grad()
    def _nucleus_pick(self, probs_row, top_p=0.9):  # probs_row: (K,V)
        sp, si = torch.sort(probs_row, dim=-1, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = cum <= top_p; keep[:, 0] = True
        sp = sp*keep; sp = sp/sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        idx = torch.multinomial(sp, 1).squeeze(-1)
        return si.gather(1, idx.unsqueeze(-1)).squeeze(-1)              # (K,)

    @torch.no_grad()
    def sample_with_prefix(self, prefix_ids, num_steps=24, top_p=0.9):
        if prefix_ids.ndim == 1: prefix_ids = prefix_ids.unsqueeze(0)
        B, pref_len = prefix_ids.size()
        L = self.max_len
        device = next(self.model.parameters()).device
        x = torch.full((B,L), self.mask_id, dtype=torch.long, device=device)
        x[:, :pref_len] = prefix_ids
        attn = (x != self.pad_id).long()

        revealed = torch.zeros(B, L, dtype=torch.bool, device=device)
        revealed[:, :pref_len] = True

        for s in range(num_steps):
            logits = self.model(x, attention_mask=attn)
            probs = torch.softmax(logits, dim=-1)
            probs[:, :, self.pad_id]  = 0
            probs[:, :, self.mask_id] = 0

            remain = (~revealed).sum(dim=1)
            k = (remain.float() / max(1, (num_steps-s))).ceil().clamp_min(1).long()

            max_prob, _ = probs.max(dim=-1)  # (B,L)
            for b in range(B):
                if remain[b] == 0: continue
                cand = torch.where(~revealed[b])[0]
                cand = cand[cand >= pref_len]
                if cand.numel() == 0: continue
                conf = max_prob[b, cand]
                topk = min(k[b].item(), cand.numel())
                chosen = cand[torch.topk(conf, topk).indices]
                picked = self._nucleus_pick(probs[b, chosen], top_p=top_p)
                x[b, chosen] = picked
                revealed[b, chosen] = True
            if revealed.all(): break
        return x

    @torch.no_grad()
    def split_candidates_after_prefix(self, seq_ids, prefix_len: int):
        gen = seq_ids[0].tolist()
        tail = gen[prefix_len:]
        out, cur = [], []
        for t in tail:
            if t == self.eos_id:
                if cur: out.append(cur); cur = []
            elif t == self.pad_id:
                continue
            else:
                cur.append(t)
        if cur: out.append(cur)
        return out