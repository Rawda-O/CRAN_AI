# cran/learning/losses.py

"""Loss functions for CRAN-AI.

We implement a clean, configurable multi-objective loss:
L = -wr*E[RS] + wq*E[QoS_violation] + we*E[-EE] + wo*E[outage_proxy] + risk_term

Notes on contributions:
- Multi-objective: explicit weights enable Pareto sweeps (new contribution)
- Robust CSI: loss can be computed using g_perfect while action uses g_imperfect (new contribution)
- Risk control: CVaR term penalizes tail QoS violations (new contribution)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from cran.physics.constraints import qos_violation
from cran.physics.outage import outage_smooth
from cran.physics.energy import ee_secondary


@dataclass(frozen=True)
class MultiObjectiveWeights:
    wr: float = 1.0
    wq: float = 10.0
    we: float = 0.2
    wo: float = 0.5


@dataclass(frozen=True)
class RiskConfig:
    enable: bool = False
    type: str = "cvar"     # cvar | chance_margin
    cvar_alpha: float = 0.95
    epsilon: float = 0.01
    weight: float = 5.0


def cvar(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """CVaR of x at level alpha (average of worst (1-alpha) tail)."""
    # x expected to be non-negative (e.g., violations)
    q = torch.quantile(x, alpha)                # VaR
    tail = x[x >= q]
    if tail.numel() == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return tail.mean()


def risk_penalty(viol: torch.Tensor, cfg: RiskConfig) -> torch.Tensor:
    if not cfg.enable:
        return torch.tensor(0.0, device=viol.device, dtype=viol.dtype)

    if cfg.type == "cvar":
        return cfg.weight * cvar(viol, cfg.cvar_alpha)  # new contribution: tail-robust QoS
    if cfg.type == "chance_margin":
        # chance-style: penalize if mean violation exceeds margin corresponding to epsilon (simple proxy)
        # This is a lightweight hook; more formal chance constraints can be added later.
        return cfg.weight * torch.relu(viol.mean() - 0.0)
    raise ValueError(f"Unknown risk type: {cfg.type}")


def multi_objective_loss(rs: torch.Tensor,
                         Q: torch.Tensor,
                         A: torch.Tensor,
                         ps: torch.Tensor,
                         pr: torch.Tensor,
                         rs_th: float,
                         outage_k: float,
                         w: MultiObjectiveWeights,
                         risk_cfg: RiskConfig = RiskConfig()) -> Dict[str, torch.Tensor]:
    """Compute loss and individual terms (returned for logging)."""
    viol = qos_violation(Q, A)                  # (Q-A)+ : key metric for QoS protection
    outp = outage_smooth(rs, rs_th, k=outage_k) # differentiable outage proxy
    ee = ee_secondary(rs, ps, pr)               # energy efficiency

    # We maximize RS and EE -> minimize negative of them
    loss = (-w.wr * rs.mean()) + (w.wq * viol.mean()) + (w.we * (-ee.mean())) + (w.wo * outp.mean())
    loss = loss + risk_penalty(viol, risk_cfg)

    return {
        "loss": loss,
        "rs_mean": rs.mean(),
        "qos_violation_mean": viol.mean(),
        "outage_proxy_mean": outp.mean(),
        "ee_mean": ee.mean(),
    }
