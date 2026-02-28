# cran/physics/csi.py

"""Perfect and imperfect CSI handling.

Key contribution supported here:
- **Robust self-supervised pairing**:
  We generate a pair (g_imperfect, g_perfect) where:
  - g_imperfect is used as the DNN input (what the system measures),
  - g_perfect is used to compute the loss during training (teacher reference),
  matching the robust training philosophy described in the attached papers.

Critical rule (prevents a known pitfall from Extended2):
- Noise is applied ONLY to channel gains, never to appended system parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class ImperfectCSIConfig:
    sigma: float = 0.15                  # estimation error strength (normalized)
    apply_to_links: Tuple[str, ...] = ("PP","PR","RP","SP")  # default: primary-sensitive links
    clamp_min: float = 0.0              # gains must stay non-negative


def make_imperfect_pair(g_perfect: Dict[str, torch.Tensor],
                        cfg: ImperfectCSIConfig) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Return (g_imperfect, g_perfect) dicts with the same keys.

    Noise model (simple, torch-native):
    g_hat = clamp(g + sigma * N(0,1))
    - additive noise is easy to control and is differentiable-friendly
    - clamp ensures physical non-negativity of gains
    """
    g_imperfect = {k: v.clone() for k, v in g_perfect.items()}  # start from perfect gains

    for lk in cfg.apply_to_links:
        if lk not in g_imperfect:
            continue
        noise = torch.randn_like(g_imperfect[lk]) * cfg.sigma  # estimation error
        g_imperfect[lk] = torch.clamp(g_imperfect[lk] + noise, min=cfg.clamp_min)  # keep g>=0

    return g_imperfect, g_perfect  # pairing: input vs loss-reference


def stack_gains(g: Dict[str, torch.Tensor], links: Tuple[str, ...]) -> torch.Tensor:
    """Stack gain dict into (batch, num_links) tensor in a fixed order."""
    return torch.stack([g[k] for k in links], dim=-1)  # fixed ordering prevents feature mismatch
