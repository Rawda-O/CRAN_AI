# cran/learning/schedulers.py

"""Learning rate schedulers (simple, stable defaults)."""

from __future__ import annotations

from typing import Optional

import torch


def build_scheduler(optimizer: torch.optim.Optimizer, name: str, max_steps: int, **kwargs):
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        # good default for smooth convergence
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    if name == "step":
        step_size = int(kwargs.get("step_size", max_steps // 3))
        gamma = float(kwargs.get("gamma", 0.5))
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    raise ValueError(f"Unknown scheduler: {name}")
