import re
import os
from pathlib import Path
import pytest

from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer

# ---------- Helpers ----------

def make_toy_vocab(tmp_path):
    vocab = tmp_path / "toy_vocab.txt"
    # Keep order stable: ids == enumerate order
    vocab.write_text("\n".join([
        "[PAD]", "[MASK]", "[EOS]", "[UNK]",
        "C","O","N","(",")","=","1","2","[NH4+]","Br","Cl","[C@H]","[Si]","/","\\","-","[n+]"
    ]), encoding="utf-8")
    return vocab

# ---------- Tests ----------

def test_vocab_special_tokens_present_and_ids(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    # presence
    assert "[PAD]" in tok.vocab and "[MASK]" in tok.vocab and "[EOS]" in tok.vocab
    # attributes map correctly
    assert tok.vocab["[PAD]"] == tok.pad_token_id
    assert tok.vocab["[MASK]"] == tok.mask_token_id
    assert tok.vocab["[EOS]"] == tok.eos_token_id
    # unk may or may not be defined; our toy has it
    assert tok.vocab.get("[UNK]", None) == tok.unk_token_id

def test_tokenize_basic_aromatics_and_ring(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    s = "C1=CC=CC=C1"
    tokens = tok.tokenize(s)
    # Expected tokenization: each char or regex-grouped token
    assert tokens == ["C","1","=","C","C","=","C","C","=","C","1"]

def test_tokenize_halogens_are_single_tokens(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    s = "CClBr"
    tokens = tok.tokenize(s)
    # 'Cl' and 'Br' should be single tokens, not 'C','l' or 'B','r'
    assert tokens == ["C","Cl","Br"]

def test_tokenize_square_bracket_group_is_single_token(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    s = "C[NH4+]C"
    tokens = tok.tokenize(s)
    assert tokens == ["C","[NH4+]","C"]

def test_encode_decode_roundtrip_without_specials(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    s = "C1=CC=CC=C1"
    ids = tok.encode(s, add_special_tokens=False, max_length=None, padding=False, truncation=False)
    # decode should give back exactly the same
    s2 = tok.decode(ids)
    assert s2 == s

def test_encode_includes_unk_for_missing_tokens(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    # Introduce a token that's not in toy vocab (e.g., aromatic 'c')
    s = "c1ccccc1"  # lowercase 'c' is absent from toy vocab
    ids = tok.encode(s, add_special_tokens=False, max_length=None, padding=False, truncation=False)
    assert tok.unk_token_id in ids

def test_padding_and_truncation(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    s = "C1=CC=CC=C1[EOS]C"
    L = 16
    ids = tok.encode(s, add_special_tokens=False, max_length=L, padding=True, truncation=True)
    assert len(ids) == L
    # ensure at least one PAD is present if sequence is shorter than L
    assert tok.pad_token_id in ids or len(tok.tokenize(s)) >= L

def test_no_space_required_equivalence(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    s1 = "C1=CC=CC=C1[EOS]C"
    s2 = "C1=CC=CC=C1 [EOS] C"
    ids1 = tok.encode(s1, add_special_tokens=False, max_length=32, padding=True, truncation=True)
    ids2 = tok.encode(s2, add_special_tokens=False, max_length=32, padding=True, truncation=True)
    assert ids1 == ids2

def test_batch_encode_decode(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    batch = ["C1=CC=CC=C1", "C[NH4+]C", "CClBr"]
    ids_batch = tok.batch_encode(batch, max_length=20, padding=True, truncation=True)
    assert isinstance(ids_batch, list) and len(ids_batch) == len(batch)
    for arr in ids_batch:
        assert len(arr) == 20
    dec = tok.batch_decode(ids_batch)
    # Decoded SMILES should be strings (pads removed)
    assert all(isinstance(x, str) for x in dec)
    assert all(len(x) > 0 for x in dec)

def test_regex_covers_supported_atoms_and_symbols(tmp_path):
    vocab = make_toy_vocab(tmp_path)
    tok = RegexSmilesTokenizer(str(vocab))
    pattern = re.compile(tok.SMI_REGEX_PATTERN)
    # Check a variety of atoms/symbols captured individually
    for s in ["[C@H]","[Si]","[n+]","/","\\","-"]:
        m = pattern.findall(s)
        assert "".join(m) == s

def test_loading_project_vocab_has_required_specials():
    # This test uses the repository's default vocab.txt
    repo_root = Path(__file__).resolve().parents[1]
    vocab_path = repo_root / "vocab.txt"
    assert vocab_path.exists(), "vocab.txt not found at project root"
    tok = RegexSmilesTokenizer(str(vocab_path))
    # must have essential tokens
    for required in ["[PAD]", "[EOS]", "[MASK]"]:
        assert required in tok.vocab
    # sanity on vocab size
    assert tok.vocab_size >= 100, "Project vocab seems unexpectedly small"
