import os, re, yaml, torch
from typing import Any, Dict

def _auto_device_str(val: str) -> str:
    return "cuda" if (val=="auto" and torch.cuda.is_available()) else ("cpu" if val=="auto" else val)

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ["train_path","valid_path","vocab_path","save_dir"]:
        if "paths" in cfg and k in cfg["paths"]:
            cfg["paths"][k] = os.path.expandvars(cfg["paths"][k])
    for sec in ("train","sample"):
        if sec in cfg and "device" in cfg[sec]:
            cfg[sec]["device"] = _auto_device_str(cfg[sec]["device"])
    return cfg

def merge_cli_overrides(cfg: Dict[str, Any], pairs):
    if not pairs: return cfg
    for kv in pairs:
        k, v = kv.split("=", 1)
        node = cfg; parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        sval = str(v)
        if sval.lower() in ("true","false"): val = sval.lower()=="true"
        elif re.fullmatch(r"-?\d+", sval):   val = int(sval)
        elif re.fullmatch(r"-?\d+\.\d*", sval) or "e" in sval.lower():
            try: val = float(sval)
            except: val = sval
        else: val = v
        node[parts[-1]] = val
    return cfg