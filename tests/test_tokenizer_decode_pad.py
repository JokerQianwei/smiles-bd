from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer

def test_decode_ignores_pad(tmp_path):
    vocab = tmp_path / "toy_vocab.txt"
    vocab.write_text("\n".join([
        "[PAD]","[MASK]","[EOS]","[UNK]","C","O"
    ]), encoding="utf-8")
    tok = RegexSmilesTokenizer(str(vocab))
    # "C[PAD]O[PAD]" should decode to "CO"
    ids = [tok.vocab["C"], tok.pad_token_id, tok.vocab["O"], tok.pad_token_id]
    s = tok.decode(ids)
    assert s == "CO"
