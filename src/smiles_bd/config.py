
import os, re
from typing import Any, Dict, Optional
import yaml
import torch

def _auto_device_str(val: str) -> str:
    if val == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return val

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # expand env vars in paths
    for k in ["train_path", "valid_path", "vocab_path", "save_dir"]:
        if "paths" in cfg and k in cfg["paths"]:
            cfg["paths"][k] = os.path.expandvars(cfg["paths"][k])
    # resolve device
    if "train" in cfg:
        cfg["train"]["device"] = _auto_device_str(cfg["train"].get("device", "auto"))
    if "sample" in cfg:
        cfg["sample"]["device"] = _auto_device_str(cfg["sample"].get("device", "auto"))
    return cfg

def merge_cli_overrides(cfg: Dict[str, Any], overrides: Optional[dict] = None) -> Dict[str, Any]:
    if not overrides:
        return cfg
    # overrides is a dict like {"model.d_model": "768", "train.lr": "1e-4"}
    for k, v in overrides.items():
        parts = k.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        # type coercion: bool -> int -> float -> str
        sval = str(v)
        if sval.lower() in ("true", "false"):
            val = sval.lower() == "true"
        elif re.fullmatch(r"-?\d+", sval):
            val = int(sval)
        elif re.fullmatch(r"-?\d+\.\d*", sval) or "e" in sval.lower():
            try:
                val = float(sval)
            except:
                val = sval
        else:
            val = v
        node[parts[-1]] = val
    return cfg
