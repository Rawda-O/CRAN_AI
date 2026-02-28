# cran/policies/robust_wrapper.py

"""Robust wrapper for self-supervised pairing training.

New contribution:
- Train with (g_hat -> action) but compute loss using g_ref in physics.
  This improves robustness to imperfect CSI without requiring labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cran.utils.tensor_utils import to_device


@dataclass(frozen=True)
class RobustBatch:
    x_imperfect: torch.Tensor   # features built from imperfect CSI (network input)
    x_perfect: torch.Tensor     # features built from perfect CSI (for loss reference if needed)
    g_imperfect: Dict[str, torch.Tensor]
    g_perfect: Dict[str, torch.Tensor]


class RobustPolicyWrapper(nn.Module):
    def __init__(self, policy: nn.Module):
        super().__init__()
        self.policy = policy

    def forward(self, x: torch.Tensor):
        return self.policy(x)

    def predict(self, x_imperfect: torch.Tensor, budgets: Dict[str, torch.Tensor]):
        # Actions are computed from imperfect observation (realistic deployment)
        return self.policy.predict(x_imperfect, budgets)
