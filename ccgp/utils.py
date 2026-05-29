"""Reproducibility and device utilities."""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    Parameters
    ----------
    seed : int
        Master seed.
    deterministic : bool
        If True, force deterministic cuDNN kernels (slower, used for the
        numerical-equivalence checks rather than the heavy training runs).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device(prefer: str | None = None):
    """Return a torch.device, preferring CUDA when available."""
    import torch

    if prefer is not None:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@contextmanager
def timer():
    """Context manager yielding a callable that returns elapsed seconds."""
    start = time.perf_counter()
    elapsed = {"t": 0.0}
    try:
        yield lambda: time.perf_counter() - start
    finally:
        elapsed["t"] = time.perf_counter() - start
