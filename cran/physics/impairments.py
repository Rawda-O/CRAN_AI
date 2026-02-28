# cran/physics/impairments.py

"""Hardware impairment hooks.

New contribution supported here:
- Residual self-interference (SI) realism for full-duplex relay.
  Papers often assume ideal cancellation; we add a tunable residual SI level (dB)
  to study practical degradation and robustness.

This file only provides helper functions; the actual SI terms are applied inside
rates_df.py / rates_cf.py to keep physics expressions centralized.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class HardwareConfig:
    enable: bool = True
    residual_si_db: float = 80.0  # higher = better cancellation; inf disables SI
    evm: float = 0.0              # optional transmitter impairment (0 ideal)


def si_linear(residual_si_db: float) -> float:
    """Convert SI suppression in dB to linear leakage factor."""
    if math.isinf(residual_si_db):
        return 0.0  # ideal cancellation (no leakage)
    return 10.0 ** (-residual_si_db / 10.0)  # leakage power ratio


def apply_residual_si(noise_power: torch.Tensor,
                      pr: torch.Tensor,
                      leakage_lin: float) -> torch.Tensor:
    """Effective noise = noise + leakage * Pr.

    This models the relay's own transmit power leaking into its receive chain.
    """
    return noise_power + leakage_lin * pr  # SI increases effective noise floor
