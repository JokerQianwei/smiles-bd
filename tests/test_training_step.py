import torch
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.data import TextLineDataset
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion

def test_training_step_runs(tmp_path, vocab_path):
    tok = RegexSmilesTokenizer(vocab_path)
    L = 64
    line = "C[SEP]" + "[PAD]"*(L-2)
    datafile = tmp_path / "train.txt"
    datafile.write_text((line + "\n") * 2, encoding="utf-8")
    ds = TextLineDataset(str(datafile), tok, max_len=L)
    # build a small batch by stacking two items
    b0, b1 = ds[0], ds[1]
    batch = {
        "input_ids": torch.stack([b0["input_ids"], b1["input_ids"]], dim=0),
        "attention_mask": torch.stack([b0["attention_mask"], b1["attention_mask"]], dim=0),
    }
    model = TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=L, dropout=0.1, tie_weights=True)
    sch = ClippedLinearSchedule(beta=0.3, omega=0.8)
    diff = MaskedDiffusion(model, tok, sch, tok.pad_token_id, tok.mask_token_id, tok.eos_token_id, max_len=L)
    out = diff.training_step(batch)
    assert torch.isfinite(out.loss).item()
    assert int(out.num_masked.item()) > 0
