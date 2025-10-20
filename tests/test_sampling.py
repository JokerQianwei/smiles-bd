import torch
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion

def test_sampling_prefix_frozen(vocab_path):
    tok = RegexSmilesTokenizer(vocab_path)
    L = 64
    model = TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=L, dropout=0.1, tie_weights=True)
    sch = ClippedLinearSchedule(beta=0.3, omega=0.8)
    diff = MaskedDiffusion(model, tok, sch, tok.pad_token_id, tok.mask_token_id, tok.eos_token_id, max_len=L)
    prefix = "C[SEP]"
    prefix_ids = torch.tensor(tok.encode(prefix, add_special_tokens=False, max_length=L, padding=False, truncation=True)).unsqueeze(0)
    seq = diff.sample_with_prefix(prefix_ids, num_steps=4, top_p=0.9)
    assert torch.equal(seq[0, :prefix_ids.size(1)], prefix_ids[0])
    assert int((seq == tok.pad_token_id).sum().item()) == 0
    assert int((seq == tok.mask_token_id).sum().item()) == 0
    cands_ids = diff.split_candidates_after_prefix(seq, prefix_len=prefix_ids.size(1))
    assert isinstance(cands_ids, list)
    for c in cands_ids:
        assert isinstance(c, list)
