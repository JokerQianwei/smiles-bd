
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer

def test_regex_smiles_tokenizer(tmp_path):
    vocab = tmp_path / "toy_vocab.txt"
    vocab.write_text("\n".join([
        "[PAD]","[MASK]","[EOS]","[UNK]","C","O","N","(",")","=","1"
    ]), encoding="utf-8")
    tok = RegexSmilesTokenizer(str(vocab))
    ids = tok.encode("C1=CC=CC=C1[EOS]C", add_special_tokens=False, max_length=32, padding=True, truncation=True)
    assert len(ids) == 32
    assert tok.pad_token_id in ids
    s = tok.decode(ids[:14])
    assert "[EOS]" in s
    # ensure no spaces are needed
    ids2 = tok.encode("C1=CC=CC=C1 [EOS] C", add_special_tokens=False, max_length=32, padding=True, truncation=True)
    assert ids == ids2
