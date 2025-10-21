import tempfile, os, torch
from smiles_bd.tokenizer import RegexSmilesTokenizer
from smiles_bd.data import TextLineDataset
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion

def write_vocab(tmp):
    vp = os.path.join(tmp, "vocab.txt")
    with open(vp, "w") as f:
        for t in ["[PAD]","[MASK]","[SEP]","[UNK]","C","O","N","1","=","(",")","Cl"]:
            f.write(t+"\n")
    return vp

def test_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        vp = write_vocab(tmp)
        tok = RegexSmilesTokenizer(vp)
        L = 64
        with open(os.path.join(tmp,"train.txt"),"w") as f:
            f.write("CO[SEP]"+"[PAD]"*(L- len("CO[SEP]".replace("[","").replace("]",""))) + "\n")
        ds = TextLineDataset(os.path.join(tmp,"train.txt"), tok, max_len=L)
        batch = {"input_ids": torch.stack([ds[0]["input_ids"]]), "attention_mask": torch.stack([ds[0]["attention_mask"]])}
        model = TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=L)
        sch = ClippedLinearSchedule()
        diff = MaskedDiffusion(model, tok, sch, tok.pad_token_id, tok.mask_token_id, tok.sep_token_id, max_len=L)
        out = diff.training_step(batch)
        assert torch.isfinite(out.loss)
        prefix = "C[SEP]"
        prefix_ids = torch.tensor(tok.encode(prefix, add_special_tokens=False, max_length=L, padding=False))
        seq = diff.sample_with_prefix(prefix_ids, num_steps=4, top_p=0.9)
        assert torch.equal(seq[0,:prefix_ids.size(0)], prefix_ids)
        cands = diff.split_candidates_after_prefix(seq, prefix_len=prefix_ids.size(0))
        assert isinstance(cands, list)
