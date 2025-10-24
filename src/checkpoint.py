import os, torch
from typing import Any, Dict

def pack_state(model, optimizer, scaler, cfg: Dict[str, Any], step: int, best_val: float, extra: Dict[str, Any] = None) -> Dict[str, Any]:
    # 取出真正的模型本体（避免 DDP / compile 包装）
    raw = getattr(model, "module", model)
    if hasattr(raw, "_orig_mod"):  # torch.compile 包装时附带的原始模块
        raw = raw._orig_mod
    # 提取干净的 state_dict（不含 _orig_mod. 或 module. 前缀）
    state = raw.state_dict()
    payload = {
        "model": state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg,
        "step": int(step),
        "best_val_loss": float(best_val),
    }
    if extra:
        payload["extra"] = extra
    return payload

def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path: str, map_location=None) -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)
