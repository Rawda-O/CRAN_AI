# cran/selection/evaluation.py

"""Evaluation helper for selector."""

from __future__ import annotations

from typing import Dict

import torch


@torch.no_grad()
def selector_accuracy(p_df: torch.Tensor, oracle_df: torch.Tensor, t: float = 0.5) -> float:
    pred = (p_df >= t).float()
    return float((pred == oracle_df).float().mean().item())
