# utils/device_guard.py
import os, torch, contextlib

# Global knobs (env-overridable)
FORCE_CPU = os.environ.get("FORCE_CPU", "0").strip() == "1"
ENC_CHUNK = int(os.environ.get("ENC_CHUNK", "1024"))
TRAIN_CHUNK = int(os.environ.get("TRAIN_CHUNK", "2048"))

# Reduce fragmentation & enable TF32 where available
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")
torch.backends.cuda.matmul.allow_tf32 = True
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False

def _cuda_self_test() -> bool:
    if not torch.cuda.is_available(): return False
    try:
        with torch.cuda.device(0):
            a = torch.randn(8,8, device="cuda")
            b = torch.randn(8,8, device="cuda")
            _ = a @ b
        empty_cuda()
        return True
    except Exception:
        return False

def pick_device():
    if FORCE_CPU: return torch.device("cpu")
    return torch.device("cuda") if _cuda_self_test() else torch.device("cpu")

def empty_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_module_device(module: torch.nn.Module, device: torch.device):
    """Move module to device if any parameter isn’t there yet."""
    try:
        p = next(module.parameters())
        if p.device.type != device.type:
            module.to(device)
    except StopIteration:
        module.to(device)

def to_device_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=(device.type=="cuda"))
    return torch.as_tensor(x, dtype=torch.float32, device=device)

@contextlib.contextmanager
def cuda_oom_fallback(to_device_fn, fallback="cpu"):
    """
    Wrap a training/encoding block; on CUDA OOM:
      - empty cache
      - move models to CPU
      - rerun the inner block once on CPU
    Use: 
      with cuda_oom_fallback(lambda dev: move_everything(dev)):
          ...training/encoding...
    """
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        empty_cuda()
        dev = torch.device(fallback)
        to_device_fn(dev)
        yield  # retry once on CPU
