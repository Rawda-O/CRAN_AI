from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional

import torch

from cran.physics.channels import LINKS


@dataclass(frozen=True)
class FeatureConfig:
    include_system: bool = False
    # Order matters for stable deployment/reproducibility
    gains_order: Tuple[str, ...] = LINKS


def build_features(g: Dict[str, torch.Tensor],
                   cfg: FeatureConfig,
                   tau: Optional[torch.Tensor] = None,
                   ps_max: Optional[torch.Tensor] = None,
                   pr_max: Optional[torch.Tensor] = None,
                   si_db: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build model input features.

    Important contribution detail:
    - System parameters (tau, budgets, SI level) are appended as *clean* inputs
      to enable generalization. They must NOT be corrupted by CSI noise.
    """
    x = torch.stack([g[k] for k in cfg.gains_order], dim=-1)  # [B, 8]
    if not cfg.include_system:
        return x

    extras = []
    if tau is not None:
        extras.append(tau.unsqueeze(-1))
    if ps_max is not None:
        extras.append(ps_max.unsqueeze(-1))
    if pr_max is not None:
        extras.append(pr_max.unsqueeze(-1))
    if si_db is not None:
        extras.append(si_db.unsqueeze(-1))
    if extras:
        x = torch.cat([x] + extras, dim=-1)
    return x
