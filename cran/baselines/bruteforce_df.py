from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from cran.physics.rates_df import DFParams, secondary_rate_df, qos_A, qos_Q_df
from cran.physics.constraints import safety_scale_joint_df


@dataclass(frozen=True)
class DFGrid:
    ps_steps: int = 21
    pr_steps: int = 21
    alpha_steps: int = 11


@torch.no_grad()
def solve_df_bruteforce(
    g: Dict[str, torch.Tensor],
    ps_max: float | None = None,
    pr_max: float | None = None,
    params: DFParams | None = None,
    grid: DFGrid = DFGrid(),
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
    a = torch.linspace(0.0, 1.0, grid.alpha_steps, device=device)

    PS, PR, A = torch.meshgrid(ps, pr, a, indexing="ij")
    PS = PS.reshape(-1)
    PR = PR.reshape(-1)
    A = A.reshape(-1)
    K = PS.numel()

    PSb = PS.unsqueeze(0).expand(B, K)
    PRb = PR.unsqueeze(0).expand(B, K)
    Ab = A.unsqueeze(0).expand(B, K)

    Aq = qos_A(g["PP"], params).unsqueeze(-1).expand(B, K)

    if safety:
        PSb, PRb = safety_scale_joint_df(g, PSb, PRb, Ab, Aq)

    rs = secondary_rate_df(g, PSb, PRb, Ab, params)
    Q  = qos_Q_df(g, PSb, PRb, Ab)

    feas = (Q <= Aq)
    rs_eff = torch.where(feas, rs, torch.full_like(rs, -1e9))

    best_idx = torch.argmax(rs_eff, dim=1)
    ar = torch.arange(B, device=device)

    return {"ps": PSb[ar, best_idx], "pr": PRb[ar, best_idx], "alpha": Ab[ar, best_idx], "rs": rs[ar, best_idx]}
