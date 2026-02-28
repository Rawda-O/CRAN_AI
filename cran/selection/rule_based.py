# cran/selection/rule_based.py

"""Rule-based DF/CF selector.

This is a strong baseline for the selector contribution:
- Evaluate both DF and CF predicted actions (or baselines), pick higher RS
  while respecting QoS after safety scaling if enabled.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from cran.physics.constraints import safety_scale_joint_df, safety_scale_joint_cf
from cran.physics.rates_df import DFParams, secondary_rate_df, qos_A, qos_Q_df
from cran.physics.rates_cf import CFParams, secondary_rate_cf, qos_A as qos_A_cf, qos_Q_cf


@torch.no_grad()
def select_best_scheme(g: Dict[str, torch.Tensor],
                       df_action: Dict[str, torch.Tensor],
                       cf_action: Dict[str, torch.Tensor],
                       df_params: DFParams,
                       cf_params: CFParams,
                       safety: bool = True) -> Dict[str, torch.Tensor]:
    device = g["PP"].device

    # DF
    ps_df, pr_df, a_df = df_action["ps"], df_action["pr"], df_action["alpha"]
    if safety:
        A_df = qos_A(g["PP"], df_params)
        ps_df, pr_df = safety_scale_joint_df(g, ps_df, pr_df, a_df, A_df)
    rs_df = secondary_rate_df(g, ps_df, pr_df, a_df, df_params)

    # CF
    ps_cf, pr_cf = cf_action["ps"], cf_action["pr"]
    if safety:
        A_cf = qos_A_cf(g["PP"], cf_params)
        ps_cf, pr_cf = safety_scale_joint_cf(g, ps_cf, pr_cf, A_cf)
    rs_cf = secondary_rate_cf(g, ps_cf, pr_cf, cf_params)

    choose_df = rs_df >= rs_cf  # maximize RS

    out = {
        "scheme": torch.where(choose_df, torch.ones_like(rs_df), torch.zeros_like(rs_df)),  # 1=DF, 0=CF
        "rs_df": rs_df,
        "rs_cf": rs_cf,
        "ps": torch.where(choose_df, ps_df, ps_cf),
        "pr": torch.where(choose_df, pr_df, pr_cf),
        "alpha": torch.where(choose_df, a_df, torch.zeros_like(a_df)),
    }
    return out
