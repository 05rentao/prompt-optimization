"""GPU memory utilities. Use between trials to reduce OOM."""
import gc
import torch


def free_gpu_memory() -> None:
    """Run GC and clear PyTorch CUDA cache. Call between trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
