import torch
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.model import TransformerDenoiser
from smiles_bd.schedule import ClippedLinearSchedule
from smiles_bd.diffusion import MaskedDiffusion
def build_tok():
    import tempfile, os
    toks=["[PAD]","[MASK]","[SEP]","[UNK]","C","O","1","=","("," )".strip(),"Cl"]
    fd,path=tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd,"w") as f:
        for t in toks: f.write(t+"\n")
    return RegexSmilesTokenizer(path)
def test_training_step_masked_only():
    tok=build_tok(); max_len=32
    model=TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=max_len)
    sch=ClippedLinearSchedule(0.3,0.8)
    diff=MaskedDiffusion(model,tok,sch,tok.pad_token_id,tok.mask_token_id,tok.sep_token_id,max_len=max_len)
    s="CO[SEP]C[SEP]" + "[PAD]"*24
    ids=tok.encode(s, add_special_tokens=False, max_length=max_len, padding=True, truncation=True)
    batch={"input_ids": torch.tensor([ids]), "attention_mask": (torch.tensor([ids])!=tok.pad_token_id).long()}
    out=diff.training_step(batch); assert torch.isfinite(out.loss); assert int(out.num_masked)>0
def test_sampling_never_remask():
    tok=build_tok(); max_len=32
    model=TransformerDenoiser(vocab_size=tok.vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=max_len)
    sch=ClippedLinearSchedule(0.3,0.8)
    diff=MaskedDiffusion(model,tok,sch,tok.pad_token_id,tok.mask_token_id,tok.sep_token_id,max_len=max_len)
    prefix="C[SEP]"
    prefix_ids=torch.tensor(tok.encode(prefix, add_special_tokens=False, max_length=max_len, padding=False, truncation=True)).unsqueeze(0)
    seq=diff.sample_with_prefix(prefix_ids, num_steps=4, top_p=0.9)
    assert torch.equal(seq[0,:prefix_ids.size(1)], prefix_ids[0])
    tail=seq[0, prefix_ids.size(1):]
    assert int((tail == tok.mask_token_id).sum()) == 0
    cands=diff.split_candidates_after_prefix(seq, prefix_len=prefix_ids.size(1)); assert isinstance(cands, list)
