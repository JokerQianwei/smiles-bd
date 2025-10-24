import os, yaml, re
from typing import Any, Dict, List

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Expand paths
    def expand(v):
        if isinstance(v, str):
            return os.path.expanduser(os.path.expandvars(v))
        if isinstance(v, dict):
            return {k: expand(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [expand(vv) for vv in v]
        return v
    return expand(cfg)

def save_config(cfg: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def merge_cli_overrides(cfg: Dict[str, Any], pairs: List[str]) -> Dict[str, Any]:
    if not pairs:
        return cfg
    for kv in pairs:
        k, v = kv.split("=", 1)
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        sval = str(v)
        if sval.lower() in ("true","false"):
            val: Any = sval.lower()=="true"
        elif re.fullmatch(r"-?\d+", sval):
            val = int(sval)
        elif re.fullmatch(r"-?\d+\.\d*", sval) or "e" in sval.lower():
            try:
                val = float(sval)
            except Exception:
                val = sval
        else:
            val = v
        node[parts[-1]] = val
    return cfg
