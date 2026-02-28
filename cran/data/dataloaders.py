from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

from cran.data.dataset import ChannelBatchGenerator
from cran.data.feature_builder import FeatureConfig, build_features
from cran.data.samplers import SystemSampler


@dataclass(frozen=True)
class LoaderConfig:
    steps_per_epoch: int = 200
    val_steps: int = 50
    generalized: bool = False
    robust: bool = False


def make_epoch_iterator(gen: ChannelBatchGenerator,
                        feat_cfg: FeatureConfig,
                        loader_cfg: LoaderConfig,
                        budgets: Dict[str, float],
                        sampler: Optional[SystemSampler] = None,
                        si_db: Optional[float] = None) -> Iterator[Dict]:
    """Yield batch dicts for one epoch."""
    device = gen.device
    B = gen.cfg.batch_size

    for _ in range(loader_cfg.steps_per_epoch):
        out = gen.sample()
        g_imp, g_ref = out["g_imperfect"], out["g_perfect"]

        if loader_cfg.generalized and sampler is not None:
            tau, ps_max, pr_max = sampler.sample(B, device)
        else:
            tau = torch.full((B,), float(budgets.get("tau", 0.5)), device=device)
            ps_max = torch.full((B,), float(budgets["Ps_max"]), device=device)
            pr_max = torch.full((B,), float(budgets["Pr_max"]), device=device)

        si = torch.full((B,), float(si_db) if si_db is not None else 0.0, device=device)

        x_imp = build_features(g_imp, feat_cfg, tau=tau if feat_cfg.include_system else None,
                               ps_max=ps_max if feat_cfg.include_system else None,
                               pr_max=pr_max if feat_cfg.include_system else None,
                               si_db=si if feat_cfg.include_system else None)

        batch = {
            "x_imperfect": x_imp,
            "g_perfect": g_ref,
            "tau": tau,
            "Ps_max": ps_max,
            "Pr_max": pr_max,
        }
        yield batch
