# cran/utils/seed.py

"""Reproducibility helpers.

Important:
- We seed python, numpy, and torch.
- We also configure cuDNN flags to control determinism vs speed.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedConfig:
    value: int = 12345
    deterministic: bool = True
    cudnn_benchmark: bool = False


def set_global_seed(cfg: SeedConfig) -> None:
    """Set seeds and deterministic flags.

    Why this matters:
    - Without consistent seeding, results vary run-to-run.
    - Determinism makes experiments reproducible (often required for papers).
    """
    os.environ["PYTHONHASHSEED"] = str(cfg.value)  # makes python hashing stable
    random.seed(cfg.value)                         # python RNG
    np.random.seed(cfg.value)                      # numpy RNG
    torch.manual_seed(cfg.value)                   # torch CPU RNG
    torch.cuda.manual_seed_all(cfg.value)          # torch GPU RNG (all devices)

    # cuDNN flags: trade speed for reproducibility
    torch.backends.cudnn.deterministic = cfg.deterministic   # deterministic convolution behavior
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark     # auto-tune kernels (faster but may vary)


def get_torch_device(prefer: str = "cuda") -> torch.device:
    """Resolve torch device.

    prefer='cuda' will use GPU if available; otherwise CPU.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")   # GPU execution path
    return torch.device("cpu")        # fallback path
