from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer

def test_vocab_contains_required_tokens(vocab_path):
    tok = RegexSmilesTokenizer(vocab_path)
    need = {"[PAD]", "[MASK]", "[SEP]", "[UNK]"}
    have = set(tok.vocab.keys())
    missing = need - have
    assert not missing, f"Missing required tokens in vocab: {missing}"
    ids = [tok.vocab[t] for t in need]
    assert len(set(ids)) == len(ids)

def test_encode_decode_padding_and_sep(vocab_path):
    tok = RegexSmilesTokenizer(vocab_path)
    L = 32
    s = "C[SEP]"
    ids = tok.encode(s, add_special_tokens=False, max_length=L, padding=True, truncation=True)
    assert len(ids) == L
    dec = tok.decode(ids)
    assert "[SEP]" in dec
    assert "[PAD]" not in dec
