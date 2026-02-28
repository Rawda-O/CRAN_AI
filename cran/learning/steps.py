from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch

from cran.physics.rates_df import DFParams, secondary_rate_df, qos_A, qos_Q_df
from cran.physics.rates_cf import CFParams, secondary_rate_cf, qos_A as qos_A_cf, qos_Q_cf
from cran.physics.constraints import safety_scale_joint_df, safety_scale_joint_cf
from cran.learning.losses import multi_objective_loss, MultiObjectiveWeights, RiskConfig


@dataclass(frozen=True)
class StepConfig:
    scheme: Literal["df", "cf"] = "df"
    safety: bool = True
    rs_threshold: float = 0.2
    outage_k: float = 10.0


def df_step_fn(model, batch: Dict, device: torch.device,
               dfp: DFParams,
               w: MultiObjectiveWeights,
               risk: RiskConfig,
               cfg: StepConfig) -> Dict[str, torch.Tensor]:
    x = batch["x_imperfect"].to(device)
    g = batch["g_perfect"]  # physics reference (perfect CSI) for robust pairing
    budgets = {"Ps_max": batch["Ps_max"].to(device), "Pr_max": batch["Pr_max"].to(device)}

    act = model.predict(x, budgets)
    ps, pr, alpha = act.ps, act.pr, act.alpha

    A = qos_A(g["PP"], dfp)
    if cfg.safety:
        ps, pr = safety_scale_joint_df(g, ps, pr, alpha, A)  # contribution: hard QoS guarantee in eval/training

    rs = secondary_rate_df(g, ps, pr, alpha, dfp)
    Q = qos_Q_df(g, ps, pr, alpha)

    loss_dict = multi_objective_loss(rs, Q, A, ps, pr, cfg.rs_threshold, cfg.outage_k, w, risk_cfg=risk)
    return loss_dict


def cf_step_fn(model, batch: Dict, device: torch.device,
               cfp: CFParams,
               w: MultiObjectiveWeights,
               risk: RiskConfig,
               cfg: StepConfig) -> Dict[str, torch.Tensor]:
    x = batch["x_imperfect"].to(device)
    g = batch["g_perfect"]
    budgets = {"Ps_max": batch["Ps_max"].to(device), "Pr_max": batch["Pr_max"].to(device)}

    act = model.predict(x, budgets)
    ps, pr = act.ps, act.pr

    A = qos_A_cf(g["PP"], cfp)
    if cfg.safety:
        ps, pr = safety_scale_joint_cf(g, ps, pr, A)

    rs = secondary_rate_cf(g, ps, pr, cfp)
    Q = qos_Q_cf(g, ps, pr)

    loss_dict = multi_objective_loss(rs, Q, A, ps, pr, cfg.rs_threshold, cfg.outage_k, w, risk_cfg=risk)
    return loss_dict
