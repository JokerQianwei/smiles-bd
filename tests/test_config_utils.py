from smiles_bd.config import merge_cli_overrides
from smiles_bd.utils import get_amp_dtype
import torch
def test_merge_cli_overrides_types():
    cfg={"a":{"b":1,"c":"str"},"flag":False}
    o=["a.b=3","flag=true","a.c=2.5","x.y=auto"]
    cfg2=merge_cli_overrides(cfg,o)
    assert cfg2["a"]["b"]==3 and isinstance(cfg2["a"]["b"],int)
    assert cfg2["flag"] is True
    assert abs(cfg2["a"]["c"] - 2.5) < 1e-6
    assert cfg2["x"]["y"]=="auto"
def test_get_amp_dtype_auto():
    dt=get_amp_dtype("auto")
    assert dt in (None, torch.bfloat16, torch.float16)
