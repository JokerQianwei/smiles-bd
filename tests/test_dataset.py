
from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
from smiles_bd.data import TextLineDataset

def test_dataset_attention_mask(tmp_path):
    vocab = tmp_path / "toy_vocab.txt"
    vocab.write_text("\n".join([
        "[PAD]","[MASK]","[EOS]","[UNK]","C","O","N","(",")","=","1"
    ]), encoding="utf-8")
    tok = RegexSmilesTokenizer(str(vocab))
    L = 64
    # concatenate without spaces and pad with literal [PAD] tokens
    line = "C[EOS]" + "[PAD]"*(L-2)
    datafile = tmp_path / "train.txt"
    datafile.write_text(line+"\n", encoding="utf-8")
    ds = TextLineDataset(str(datafile), tok, max_len=L)
    ex = ds[0]
    assert ex["input_ids"].shape[0] == L
    assert ex["attention_mask"][0] == 1  # 'C'
    assert ex["attention_mask"][1] == 1  # 'EOS'
    assert ex["attention_mask"][2:].sum() == 0  # PADs
