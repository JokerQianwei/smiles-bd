
import os
import torch
from typing import Any, Dict

def save_checkpoint(module, path: str, meta: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": module.state_dict(), "meta": meta or {}}
    torch.save(payload, path)

def load_checkpoint(module, path: str, map_location=None):
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        module.load_state_dict(payload["state_dict"])
        return payload.get("meta", {})
    else:
        module.load_state_dict(payload)
        return {}

def peek_meta(path: str, map_location=None):
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict):
        return payload.get("meta", {})
    return {}

def set_seed(seed: int = 42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
