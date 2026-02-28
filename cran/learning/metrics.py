# cran/learning/metrics.py

"""Metrics helpers (torch-native)."""

from __future__ import annotations

from typing import Dict

import torch

from cran.physics.constraints import qos_violation
from cran.physics.outage import outage_hard
from cran.physics.energy import ee_secondary


@torch.no_grad()
def compute_metrics(rs: torch.Tensor, Q: torch.Tensor, A: torch.Tensor,
                    ps: torch.Tensor, pr: torch.Tensor, rs_th: float) -> Dict[str, float]:
    viol = qos_violation(Q, A)
    out = outage_hard(rs, rs_th)
    ee = ee_secondary(rs, ps, pr)

    return {
        "rs_mean": float(rs.mean().item()),
        "qos_violation_mean": float(viol.mean().item()),
        "outage_prob": float(out.mean().item()),
        "ee_mean": float(ee.mean().item()),
    }
