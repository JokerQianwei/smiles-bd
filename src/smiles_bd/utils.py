
import os, json, contextlib
from typing import Any, Dict, Optional
import torch
import torch.distributed as dist

def save_checkpoint(module, path: str, meta: Dict[str, Any] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"state_dict": getattr(module, "module", module).state_dict(), "meta": meta or {}}
    torch.save(payload, path)

def load_checkpoint(module, path: str, map_location=None):
    payload = torch.load(path, map_location=map_location)
    sd = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
    getattr(module, "module", module).load_state_dict(sd)
    return payload.get("meta", {}) if isinstance(payload, dict) else {}

def peek_meta(path: str, map_location=None):
    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict):
        return payload.get("meta", {})
    return {}

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_torch_backends():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

# ----- Distributed helpers -----
def distributed_init() -> Dict[str, Any]:
    # Initialize torch.distributed if launched with torchrun
    if dist.is_available() and not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                backend = "nccl"
            else:
                backend = "gloo"
            dist.init_process_group(backend=backend)
            return {"distributed": True, "rank": rank, "world_size": world_size, "local_rank": local_rank}
    return {"distributed": False, "rank": 0, "world_size": 1, "local_rank": 0}

def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    if is_distributed():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

# ----- SDPA backend context -----
@contextlib.contextmanager
def sdpa_kernel_ctx(backend: str = "auto"):
    """
    backend: 'auto'|'math'|'mem_efficient'|'flash'
    """
    if not torch.cuda.is_available():
        yield
        return
    from torch.backends.cuda import sdp_kernel
    if backend == "auto":
        with sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            yield
    elif backend == "math":
        with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            yield
    elif backend == "mem_efficient":
        with sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            yield
    elif backend == "flash":
        with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            yield
    else:
        yield

# ----- AMP helpers -----
def get_amp_dtype(mode: str = "auto"):
    mode = (mode or "auto").lower()
    if mode == "off":
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_available()):
        # heuristic: bf16 if device compute capability >=8.0
        try:
            major, minor = torch.cuda.get_device_capability()
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
    if mode in ("fp16","half") or mode == "auto":
        return torch.float16
    return None

@contextlib.contextmanager
def autocast_ctx(amp_dtype):
    if amp_dtype is None:
        yield
    else:
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=torch.cuda.is_available()):
            yield

def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)
