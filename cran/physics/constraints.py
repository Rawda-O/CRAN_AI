# cran/physics/constraints.py

"""Constraints and safety post-processing.

New contribution supported here:
- **Safety-Guarantee Layer**: a lightweight post-processing that enforces
  the primary QoS constraint at inference time, even if the DNN output violates it.

Why it matters:
- Penalty-based training (lambda) reduces violations but does not guarantee them.
- Safety layer provides a hard guarantee by scaling powers to satisfy Q <= A.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from cran.utils.math_utils import clamp_unit_interval, safe_sqrt
from cran.physics.rates_cf import qos_Q_cf
from cran.physics.rates_df import qos_Q_df


@dataclass(frozen=True)
class SafetyConfig:
    enable: bool = True
    mode: str = "scale_joint"   # currently only scale_joint (guarantee-friendly)


def clamp_powers(ps: torch.Tensor, pr: torch.Tensor, ps_max: float, pr_max: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clamp powers to feasible box constraints."""
    ps = torch.clamp(ps, 0.0, ps_max)  # transmit power must be within [0, Pmax]
    pr = torch.clamp(pr, 0.0, pr_max)
    return ps, pr

def _apply_safety_margin(A: torch.Tensor, rel_margin: float = 1e-6) -> torch.Tensor:
    """Conservative safety margin for numerical robustness.

    Use a tiny *relative* margin so that safety scaling targets A_eff <= A.
    This mitigates floating-point roundoff where Q' may slightly exceed A.
    """
    rel_margin = float(rel_margin)
    if rel_margin <= 0:
        return A
    return torch.clamp(A * (1.0 - rel_margin), min=0.0)



def safety_scale_joint_cf(g: Dict[str, torch.Tensor],
                          ps: torch.Tensor,
                          pr: torch.Tensor,
                          A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safety scaling for CF: Q is affine => scaling guarantees feasibility.

    If Q = gSP*Ps + gRP*Pr, then scaling both Ps and Pr by t scales Q by t.
    Choose t = min(1, A/Q).
    """
    Q = qos_Q_cf(g, ps, pr)
    A_eff = _apply_safety_margin(A)
    t = torch.minimum(torch.ones_like(Q), A_eff / Q.clamp_min(1e-12))  # ensure Q'<=A
    return t * ps, t * pr


def safety_scale_joint_df(g: Dict[str, torch.Tensor],
                          ps: torch.Tensor,
                          pr: torch.Tensor,
                          alpha: torch.Tensor,
                          A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Safety scaling for DF.

    Q_df = gSP*Ps + gRP*Pr + 2*alpha*sqrt(gSP*gRP*Ps*Pr)
    Under joint scaling Ps<-tPs, Pr<-tPr, the cross term scales by t as well,
    so Q_df scales linearly with t. Therefore the same t = min(1, A/Q) guarantees feasibility.
    """
    Q = qos_Q_df(g, ps, pr, alpha)
    A_eff = _apply_safety_margin(A)
    t = torch.minimum(torch.ones_like(Q), A_eff / Q.clamp_min(1e-12))  # hard guarantee: Q'<=A
    return t * ps, t * pr


def qos_violation(Q: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Positive part of constraint violation (useful for penalties/metrics)."""
    return torch.relu(Q - A)  # 0 when feasible

def qos_violation_tolerant(Q: torch.Tensor, A: torch.Tensor, rel_tol: float = 1e-6, abs_tol: float = 0.0) -> torch.Tensor:
    """Positive-part violation with numeric tolerance (for metrics).

    Treats Q <= A*(1+rel_tol) + abs_tol as feasible.
    """
    thr = A * (1.0 + float(rel_tol)) + float(abs_tol)
    return torch.relu(Q - thr)


def qos_violation_prob(Q: torch.Tensor, A: torch.Tensor, rel_tol: float = 1e-6, abs_tol: float = 0.0) -> torch.Tensor:
    """Probability of violation with tolerance."""
    v = qos_violation_tolerant(Q, A, rel_tol=rel_tol, abs_tol=abs_tol)
    return (v > 0).float().mean()




def enforce_alpha_bounds(alpha: torch.Tensor) -> torch.Tensor:
    """Alpha must be within [0,1] (DF-specific)."""
    return clamp_unit_interval(alpha)
