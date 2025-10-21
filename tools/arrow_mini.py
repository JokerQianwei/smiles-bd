import pyarrow as pa
import pyarrow.ipc as ipc
from smiles_bd.tokenizer import RegexSmilesTokenizer
import numpy as np
tok = RegexSmilesTokenizer("vocab.txt")

# 使用绝对路径
file_path = '/data/yqw/bd3lms-alpha/data/DrugLikeSMILSE_packed1024_butina_s055_blocks10k/train/data-01615-of-01616.arrow'

with open(file_path, "rb") as f:
    try:
        reader = ipc.open_file(f)
    except pa.ArrowInvalid:
        f.seek(0)
        reader = ipc.open_stream(f)
    
    df = reader.read_all().to_pandas()

print(df.head())
print("----------------")
print(df['text'][:5])
print("----------------")
print(tok.encode(df['input'][0]))
