# cran/policies/df_policy.py

"""DF policy network.

Outputs (new AI-native action space):
- gamma_ps in [0,1]  -> Ps = gamma_ps * Ps_max
- gamma_pr in [0,1]  -> Pr = gamma_pr * Pr_max
- alpha    in [0,1]  -> correlation / superposition parameter in DF expression

Contribution note:
- Using normalized gammas is a key enabler for **generalization** across power budgets.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from cran.policies.base_policy import BasePolicy, PolicyIO
from cran.utils.math_utils import clamp_unit_interval


class DFPolicy(BasePolicy):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (256,256,128), dropout: float = 0.0):
        super().__init__(in_dim=in_dim, out_dim=3, hidden=hidden, dropout=dropout)  # ps, pr, alpha

    def predict(self, x: torch.Tensor, budgets: Dict[str, torch.Tensor]) -> PolicyIO:
        y = self.forward(x)
        gamma_ps = torch.sigmoid(y[..., 0])  # sigmoid -> [0,1] (stable)
        gamma_pr = torch.sigmoid(y[..., 1])
        alpha    = torch.sigmoid(y[..., 2])  # alpha bounds are critical for DF feasibility

        ps = gamma_ps * budgets["Ps_max"]
        pr = gamma_pr * budgets["Pr_max"]

        return PolicyIO(ps=ps, pr=pr, alpha=alpha, gamma_ps=gamma_ps, gamma_pr=gamma_pr)
