# cran/policies/base_policy.py

"""Base policy interface.

Key design choices (important for 'from scratch' quality):
- Outputs are **normalized** (gamma in [0,1]) then mapped to physical powers using budgets.
  This enables **generalization** across Ps_max/Pr_max (new contribution).
- All code is torch-native and GPU-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cran.utils.math_utils import clamp_unit_interval
from cran.utils.tensor_utils import ensure_float32


@dataclass(frozen=True)
class PolicyIO:
    """Standard policy output container."""
    ps: torch.Tensor
    pr: torch.Tensor
    alpha: Optional[torch.Tensor] = None
    # normalized outputs (optional for logging / selector inputs)
    gamma_ps: Optional[torch.Tensor] = None
    gamma_pr: Optional[torch.Tensor] = None


class MLP(nn.Module):
    """Simple MLP backbone (clean + deployable)."""
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (256, 256, 128), out_dim: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BasePolicy(nn.Module):
    """Base class for DF/CF policies."""
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256,256,128), dropout: float = 0.0):
        super().__init__()
        self.backbone = MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = ensure_float32(x)  # keep float32 for stable GPU performance
        return self.backbone(x)

    @staticmethod
    def _map_gamma_to_power(gamma: torch.Tensor, p_max: torch.Tensor) -> torch.Tensor:
        """Map gamma in [0,1] to physical power using budget p_max."""
        gamma = clamp_unit_interval(gamma)  # enforce [0,1] physically
        return gamma * p_max                # scaling is differentiable

    def predict(self, x: torch.Tensor, budgets: Dict[str, torch.Tensor]) -> PolicyIO:
        raise NotImplementedError
