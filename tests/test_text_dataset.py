from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.data import TextLineDataset

def test_textline_attention_mask(tmp_path, vocab_path):
    tok = RegexSmilesTokenizer(vocab_path)
    L = 64
    line = "C[SEP]" + "[PAD]"*(L-2)
    datafile = tmp_path / "train.txt"
    datafile.write_text(line + "\n", encoding="utf-8")
    ds = TextLineDataset(str(datafile), tok, max_len=L)
    ex = ds[0]
    assert ex["input_ids"].shape[0] == L
    assert int(ex["attention_mask"][0].item()) == 1
    assert int(ex["attention_mask"][1].item()) == 1
    assert int(ex["attention_mask"][2:].sum().item()) == 0
