import tempfile, os
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
def _write_vocab(tokens):
    fd,path=tempfile.mkstemp(suffix=".txt")
    with os.fdopen(fd,"w") as f:
        for t in tokens: f.write(t+"\n")
    return path
def test_tokenizer_encode_decode_sep():
    vocab=["[PAD]","[MASK]","[SEP]","[UNK]","C","O","1","=","("," )".strip(),"Cl"]
    vp=_write_vocab(vocab); tok=RegexSmilesTokenizer(vp)
    s="C1=CC=CC=C1[SEP]C"
    ids=tok.encode(s, add_special_tokens=False, max_length=32, padding=True, truncation=True)
    assert len(ids)==32
    s2=tok.decode(ids)
    assert "[PAD]" not in s2 and "[SEP]" in s2
