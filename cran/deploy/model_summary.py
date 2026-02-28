# cran/deploy/model_summary.py

"""Model summary helper.

We keep it torch-only to avoid extra dependencies.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


@torch.no_grad()
def model_size_mb(model: torch.nn.Module) -> float:
    # Rough estimate: float32 parameters => 4 bytes
    n = sum(p.numel() for p in model.parameters())
    return (n * 4.0) / (1024.0 ** 2)
