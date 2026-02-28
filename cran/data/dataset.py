from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig
from cran.physics.csi import ImperfectCSIConfig, make_imperfect_pair


@dataclass(frozen=True)
class ChannelBatchConfig:
    batch_size: int = 2048
    channel_model: str = "gaussian"
    mixed_weights: Optional[Dict[str, float]] = None
    # geometry/fading are dict-compatible with GeometryConfig/FadingConfig
    geometry: Optional[dict] = None
    fading: Optional[dict] = None


class ChannelBatchGenerator:
    """On-the-fly batch generator (GPU-first).

    Returns gains for:
    - imperfect observation (for policy input) if robust enabled
    - perfect reference (for physics loss) if robust enabled
    """

    def __init__(self,
                 cfg: ChannelBatchConfig,
                 device: torch.device,
                 robust: bool = False,
                 imperfect_cfg: Optional[ImperfectCSIConfig] = None):
        self.cfg = cfg
        self.device = device
        self.robust = robust
        self.imperfect_cfg = imperfect_cfg

        geo = GeometryConfig(**(cfg.geometry or {}))
        fad = FadingConfig(**(cfg.fading or {}))
        self.geo = geo
        self.fad = fad

    @torch.no_grad()
    def sample(self) -> Dict[str, Dict[str, torch.Tensor]]:
        g = generate_gains(self.cfg.batch_size, self.cfg.channel_model, self.device,
                           geo=self.geo, fad=self.fad, mixed_weights=self.cfg.mixed_weights)
        if self.robust and self.imperfect_cfg is not None:
            g_hat, g_ref = make_imperfect_pair(g, self.imperfect_cfg)
            return {"g_imperfect": g_hat, "g_perfect": g_ref}
        return {"g_imperfect": g, "g_perfect": g}
