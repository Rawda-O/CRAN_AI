# cran/policies/cf_policy.py

"""CF policy network.

Outputs:
- gamma_ps in [0,1] -> Ps
- gamma_pr in [0,1] -> Pr

No alpha for CF (by design).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from cran.policies.base_policy import BasePolicy, PolicyIO


class CFPolicy(BasePolicy):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (256,256,128), dropout: float = 0.0):
        super().__init__(in_dim=in_dim, out_dim=2, hidden=hidden, dropout=dropout)

    def predict(self, x: torch.Tensor, budgets: Dict[str, torch.Tensor]) -> PolicyIO:
        y = self.forward(x)
        gamma_ps = torch.sigmoid(y[..., 0])
        gamma_pr = torch.sigmoid(y[..., 1])

        ps = gamma_ps * budgets["Ps_max"]
        pr = gamma_pr * budgets["Pr_max"]

        return PolicyIO(ps=ps, pr=pr, alpha=None, gamma_ps=gamma_ps, gamma_pr=gamma_pr)
