
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F

@dataclass
class TrainOutput:
    loss: torch.Tensor
    token_nll: torch.Tensor
    num_masked: torch.Tensor

class MaskedDiffusion(torch.nn.Module):
    """
    Full-sequence masked diffusion (block_size == length), SUBS parameterization.
    """
    def __init__(self, model, tokenizer, schedule, pad_token_id: int, mask_token_id: int, eos_token_id: int, max_len: int = 1024):
        super().__init__()
        self.model = model
        self.tok = tokenizer
        self.schedule = schedule
        self.pad_id = pad_token_id
        self.mask_id = mask_token_id
        self.eos_id = eos_token_id
        self.max_len = max_len

    @torch.no_grad()
    def _make_prefix_init(self, prefix_ids: torch.Tensor, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        L = self.max_len
        if prefix_ids.ndim == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        assert prefix_ids.size(0) == batch_size
        pref_len = prefix_ids.size(1)
        if pref_len > L:
            raise ValueError(f"Prefix length {pref_len} exceeds model length {L}")
        x = torch.full((batch_size, L), fill_value=self.mask_id, dtype=torch.long, device=prefix_ids.device)
        x[:, :pref_len] = prefix_ids
        attn = torch.ones_like(x, dtype=torch.long)
        return x, attn

    def _mask_with_rate(self, x0: torch.Tensor, attn: torch.Tensor, mask_rate: torch.Tensor):
        B, L = x0.shape
        probs = mask_rate.expand(B, 1).repeat(1, L)
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
        mask_rate = self.schedule.sample_mask_rate((B, 1), device=device)
        xt, mask_positions = self._mask_with_rate(x0, attn, mask_rate)
        logits = self.model(xt, attention_mask=attn)
        # SUBS constraints: disallow emitting [MASK]/[PAD]
        logits[:,:,self.mask_id] = -1e9
        logits[:,:,self.pad_id] = -1e9
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), x0.view(-1), reduction="none").view(B, L)
        masked_loss = (ce * mask_positions.float())
        weights = self.schedule.loss_weight(mask_rate).view(B, 1)
        loss = (masked_loss.sum(dim=1) / mask_positions.sum(dim=1).clamp_min(1)).mul(weights.squeeze(1)).mean()
        token_nll = (masked_loss.sum() / mask_positions.sum().clamp_min(1)).detach()
        return TrainOutput(loss=loss, token_nll=token_nll, num_masked=mask_positions.sum())

    @torch.no_grad()
    def nucleus(self, probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
        B, L, V = probs.shape
        flat = probs.reshape(B*L, V)
        sp, si = torch.sort(flat, dim=-1, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        keep = (cum <= p).float()
        keep[:, 0] = 1.0
        sp = sp * keep
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        idx = torch.multinomial(sp, 1).squeeze(-1)
        picked = si.gather(1, idx.unsqueeze(-1)).squeeze(-1)
        return picked.view(B, L)

    @torch.no_grad()
    def sample_with_prefix(self, prefix_ids: torch.Tensor, num_steps: int = 24, top_p: float = 0.9) -> torch.Tensor:
        if prefix_ids.ndim == 1:
            prefix_ids = prefix_ids.unsqueeze(0)
        prefix_ids = prefix_ids.to(next(self.parameters()).device)
        B = prefix_ids.size(0)
        x, attn = self._make_prefix_init(prefix_ids, batch_size=B)
        pref_len = prefix_ids.size(1)

        revealed = torch.zeros(B, self.max_len, dtype=torch.bool, device=x.device)
        revealed[:, :pref_len] = True

        for s in range(num_steps):
            logits = self.model(x, attention_mask=attn)
            # forbid [MASK]/[PAD]
            logits[:,:,self.mask_id] = -1e9
            logits[:,:,self.pad_id]   = -1e9
            probs = torch.softmax(logits, dim=-1)

            masked_positions = (x == self.mask_id)
            if masked_positions.sum() == 0:
                break

            remain = masked_positions.sum(dim=1)
            k = (remain.float() / max(1, (num_steps - s))).ceil().clamp_min(1).long()

            max_prob, _ = probs.max(dim=-1)
            for b in range(B):
                if remain[b] == 0:
                    continue
                cand_idx = torch.where(masked_positions[b])[0]
                cand_idx = cand_idx[cand_idx >= pref_len]  # never touch prefix
                if cand_idx.numel() == 0:
                    continue
                conf = max_prob[b, cand_idx]
                topk = min(k[b].item(), cand_idx.numel())
                chosen = cand_idx[torch.topk(conf, topk).indices]
                chosen_probs = probs[b, chosen]
                chosen_ids = self.nucleus(chosen_probs.unsqueeze(0), p=top_p).squeeze(0)
                x[b, chosen] = chosen_ids
                revealed[b, chosen] = True

        return x

    @torch.no_grad()
    def split_candidates_after_prefix(self, seq_ids: torch.Tensor, prefix_len: int) -> List[List[int]]:
        gen = seq_ids.squeeze(0).tolist()
        gen_tail = gen[prefix_len:]
        out, cur = [], []
        for t in gen_tail:
            if t == self.eos_id:
                if len(cur) > 0:
                    out.append(cur)
                cur = []
            elif t == self.pad_id:
                continue
            else:
                cur.append(t)
        if len(cur) > 0:
            out.append(cur)
        return out
