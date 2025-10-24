import pyarrow as pa
import pyarrow.ipc as ipc
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tokenizer import RegexSmilesTokenizer
import numpy as np
tok = RegexSmilesTokenizer("vocab.txt")

# 使用绝对路径
file_path = '/data/yqw/smiles-bd/cache/cache-DrugLikeSMILSE-debug/train/data-00000-of-00001.arrow'
with open(file_path, "rb") as f:
    try:
        reader = ipc.open_file(f)
    except pa.ArrowInvalid:
        f.seek(0)
        reader = ipc.open_stream(f)
    
    df = reader.read_all().to_pandas()

print(df.head())
print("----------------")
