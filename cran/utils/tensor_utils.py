# cran/utils/tensor_utils.py

"""Tensor helpers for GPU-first execution."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch


def to_device(x: Any, device: torch.device) -> Any:
    """Move tensors (and nested containers of tensors) to device."""
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)   # non_blocking helps when pin_memory=True
    if isinstance(x, Mapping):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


def assert_on_device(x: Any, device: torch.device, name: str = "tensor") -> None:
    """Fail fast if a tensor is not on the intended device."""
    if isinstance(x, torch.Tensor):
        if x.device != device:
            raise RuntimeError(f"{name} is on {x.device} but expected {device}")  # prevent silent CPU fallback
    elif isinstance(x, Mapping):
        for k, v in x.items():
            assert_on_device(v, device, f"{name}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            assert_on_device(v, device, f"{name}[{i}]")


def detach_to_cpu(x: torch.Tensor):
    """Detach and move to CPU for plotting/logging."""
    return x.detach().cpu().numpy()  # no grads, no GPU dependency


def ensure_float32(x: torch.Tensor) -> torch.Tensor:
    """Force float32 for stable GPU performance."""
    return x.float()  # float32 is the default fast dtype for most GPU kernels
