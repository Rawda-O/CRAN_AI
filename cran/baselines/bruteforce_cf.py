from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from cran.physics.rates_cf import CFParams, secondary_rate_cf, qos_A, qos_Q_cf
from cran.physics.constraints import safety_scale_joint_cf


@dataclass(frozen=True)
class CFGrid:
    ps_steps: int = 41
    pr_steps: int = 41


@torch.no_grad()
def solve_cf_bruteforce(
    g: Dict[str, torch.Tensor],
    ps_max: float | None = None,
    pr_max: float | None = None,
    params: CFParams | None = None,
    grid: CFGrid = CFGrid(),
    safety: bool = True,
    Ps_max: float | None = None,
    Pr_max: float | None = None,
) -> Dict[str, torch.Tensor]:
    assert params is not None, "params is required"
    Ps_max = float(ps_max if ps_max is not None else Ps_max)
    Pr_max = float(pr_max if pr_max is not None else Pr_max)

    device = g["PP"].device
    B = g["PP"].shape[0]

    ps = torch.linspace(0.0, Ps_max, grid.ps_steps, device=device)
    pr = torch.linspace(0.0, Pr_max, grid.pr_steps, device=device)

    PS, PR = torch.meshgrid(ps, pr, indexing="ij")
    PS = PS.reshape(-1)  # [K]
    PR = PR.reshape(-1)
    K = PS.numel()

    PSb = PS.unsqueeze(0).expand(B, K)
    PRb = PR.unsqueeze(0).expand(B, K)

    Aq = qos_A(g["PP"], params).unsqueeze(-1).expand(B, K)

    if safety:
        PSb, PRb = safety_scale_joint_cf(g, PSb, PRb, Aq)

    rs = secondary_rate_cf(g, PSb, PRb, params)  # [B,K]
    Q  = qos_Q_cf(g, PSb, PRb)                   # [B,K]

    feas = (Q <= Aq)
    rs_eff = torch.where(feas, rs, torch.full_like(rs, -1e9))

    best_idx = torch.argmax(rs_eff, dim=1)
    ar = torch.arange(B, device=device)

    return {"ps": PSb[ar, best_idx], "pr": PRb[ar, best_idx], "rs": rs[ar, best_idx]}
