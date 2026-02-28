# cran/selection/threshold_tuning.py

"""Threshold tuning for selector probability.

We sweep thresholds and pick the best according to mean RS (or other metric).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


@torch.no_grad()
def tune_threshold(p_df: torch.Tensor,
                   rs_df: torch.Tensor,
                   rs_cf: torch.Tensor,
                   grid: List[float]) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best_rs = -1e9
    stats = {}

    for t in grid:
        choose_df = p_df >= t
        rs = torch.where(choose_df, rs_df, rs_cf).mean().item()
        stats[str(t)] = rs
        if rs > best_rs:
            best_rs = rs
            best_t = t

    return best_t, {"best_rs": best_rs, "grid_rs": stats}
