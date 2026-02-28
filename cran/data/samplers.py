from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import torch


@dataclass(frozen=True)
class SystemSampler:
    """Sampler for system parameters to support generalization experiments."""
    tau_range: Tuple[float, float] = (0.1, 0.9)
    ps_max_range: Tuple[float, float] = (1.0, 10.0)
    pr_max_range: Tuple[float, float] = (1.0, 10.0)

    def sample(self, batch: int, device: torch.device):
        tau = torch.empty(batch, device=device).uniform_(*self.tau_range)
        ps_max = torch.empty(batch, device=device).uniform_(*self.ps_max_range)
        pr_max = torch.empty(batch, device=device).uniform_(*self.pr_max_range)
        return tau, ps_max, pr_max
