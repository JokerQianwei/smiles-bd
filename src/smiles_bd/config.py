import os, yaml
from typing import Any, Dict

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

def merge_cli_overrides(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    # shallow-merge; nested dictionaries are updated recursively
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
        return a
    return merge(dict(cfg), overrides)
