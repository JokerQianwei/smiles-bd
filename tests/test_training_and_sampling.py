
import torch
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion
from smiles_bd.data import TextLineDataset

def make_vocab(tmp_path):
    vocab = tmp_path / "toy_vocab.txt"
    vocab.write_text("\n".join([
        "[PAD]","[MASK]","[EOS]","[UNK]","C","O","N","(",")","=","1"
    ]), encoding="utf-8")
    return str(vocab)

def test_training_step_and_sampling(tmp_path):
    vocab_path = make_vocab(tmp_path)
    tok = RegexSmilesTokenizer(vocab_path)
    L = 64
    line = "CC[EOS]" + "[PAD]"*(L-3)
    trainf = tmp_path / "train.txt"
    validf = tmp_path / "valid.txt"
    trainf.write_text((line+"\n")*4, encoding="utf-8")
    validf.write_text((line+"\n")*2, encoding="utf-8")

    ds = TextLineDataset(str(trainf), tok, max_len=L)
    batch = {"input_ids": torch.stack([ds[0]["input_ids"], ds[1]["input_ids"]]),
             "attention_mask": torch.stack([ds[0]["attention_mask"], ds[1]["attention_mask"]])}

    model = TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=L)
    schedule = ClippedLinearSchedule(beta=0.3, omega=0.8)
    diffuser = MaskedDiffusion(model, tok, schedule, pad_token_id=tok.pad_token_id, mask_token_id=tok.mask_token_id, eos_token_id=tok.eos_token_id, max_len=L)

    out = diffuser.training_step(batch)
    assert torch.isfinite(out.loss)
    # prefix sampling: prefix kept
    prefix_ids = torch.tensor(tok.encode("C", add_special_tokens=False, max_length=L, padding=False))
    gen = diffuser.sample_with_prefix(prefix_ids, num_steps=4, top_p=0.9)
    assert gen.shape[1] == L
    assert gen[0,0].item() == tok.vocab["C"]
    # SUBS constraint: avoid PAD after prefix
    assert (gen[0,1:] != tok.pad_token_id).all()
    # split candidates
    cands = diffuser.split_candidates_after_prefix(gen, prefix_len=1)
    assert isinstance(cands, list)
