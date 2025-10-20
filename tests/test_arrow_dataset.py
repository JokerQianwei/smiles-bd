import pytest

pytestmark = pytest.mark.arrow

def _has_pyarrow():
    try:
        import pyarrow as pa  # noqa
        import pyarrow.ipc as pa_ipc  # noqa
        return True
    except Exception:
        return False

@pytest.mark.skipif(not _has_pyarrow(), reason="pyarrow not installed")
def test_arrow_dataset_roundtrip(tmp_path, vocab_path):
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    from smiles_bd.tokenizer_smiles import RegexSmilesTokenizer
    from smiles_bd.data import ArrowDataset

    tok = RegexSmilesTokenizer(vocab_path)
    L = 64
    line = "C[SEP]" + "[PAD]"*(L-2)
    table = pa.table({"input": [line, line]})
    arrow_file = tmp_path / "data.arrow"
    with pa_ipc.new_file(str(arrow_file), table.schema) as writer:
        writer.write(table)
    ds = ArrowDataset(str(arrow_file), tok, max_len=L)
    ex = ds[0]
    assert ex["input_ids"].shape[0] == L
    assert int(ex["attention_mask"][0].item()) == 1
    assert int(ex["attention_mask"][1].item()) == 1
