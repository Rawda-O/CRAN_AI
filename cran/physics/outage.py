# cran/physics/outage.py

"""Outage metrics (secondary)."""

from __future__ import annotations

import torch

from cran.utils.math_utils import outage_indicator, smooth_outage_proxy


def outage_hard(rs: torch.Tensor, rs_th: float) -> torch.Tensor:
    """Hard outage probability indicator per-sample."""
    return outage_indicator(rs, rs_th)


def outage_smooth(rs: torch.Tensor, rs_th: float, k: float = 10.0) -> torch.Tensor:
    """Differentiable outage proxy (for multi-objective training)."""
    return smooth_outage_proxy(rs, rs_th, k=k)
