import tempfile, os, torch
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.data import TextLineDataset
def _write_vocab(tokens):
    fd,path=tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd,"w") as f:
        for t in tokens: f.write(t+"\n")
    return path
def _write_lines(lines):
    fd,path=tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd,"w") as f:
        for ln in lines: f.write(ln+"\n")
    return path
def test_text_dataset_mask_and_len():
    vocab=["[PAD]","[MASK]","[SEP]","[UNK]","C","O"]
    vp=_write_vocab(vocab); tok=RegexSmilesTokenizer(vp)
    line="CO[SEP]C[SEP]"
    path=_write_lines([line + "[PAD]"*24])
    ds=TextLineDataset(path, tok, max_len=32)
    ex=ds[0]; ids,attn=ex["input_ids"], ex["attention_mask"]
    assert ids.shape[0]==32
    assert int(attn.sum()) == int((ids != tok.pad_token_id).sum())
