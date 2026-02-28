# cran/physics/rates_cf.py

"""Compress-and-Forward (CF) achievable rates and QoS constraint.

Implemented from the CF expressions extracted from the attached paper text:
RS = C( (K1*gRS*PS*PR + gSS*PS*(K1*PS + K2)) / (K2*gRS*PR + \tilde{N}_S*(K1*PS + K2)) )
Q  = gSP*PS + gRP*PR  (affine QoS constraint under CF)

New contributions enabled here:
- Supports mixed channel training (generalization contribution)
- Supports robust pairing (CSI module)
- Supports safety layer (constraints module)
- Supports residual SI realism (impairments module) through \tilde{N}_R if enabled

Important implementation detail:
- All functions are **batch-compatible** and also support an extra grid dimension
  (e.g., ps shaped [B,K]) for brute-force baselines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from cran.utils.math_utils import capacity_awgn, safe_sqrt
from cran.physics.impairments import apply_residual_si, si_linear, HardwareConfig


def _bcast_link(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Broadcast gains g shaped [B] to match a reference tensor x shaped [B,...]."""
    while g.ndim < x.ndim:
        g = g.unsqueeze(-1)
    return g


@dataclass(frozen=True)
class CFParams:
    tau: float = 0.25
    noise_power: float = 1.0
    Pp: float = 10.0
    Rp0: float = 1.0
    rho_z: float = 0.0  # noise correlation coefficient (0 = independent noises)


def effective_noises(
    g: Dict[str, torch.Tensor],
    p: CFParams,
    pr: torch.Tensor,
    hw: HardwareConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute \tilde{N}_R and \tilde{N}_S used in CF formulas."""
    n0 = torch.tensor(p.noise_power, device=pr.device, dtype=pr.dtype)

    gPR = _bcast_link(g["PR"], pr)
    gPS = _bcast_link(g["PS"], pr)

    nR = n0 + p.Pp * gPR  # primary->relay interference
    nS = n0 + p.Pp * gPS  # primary->secondary interference

    if hw.enable:
        leakage = si_linear(hw.residual_si_db)  # residual SI knob (new contribution)
        nR = apply_residual_si(nR, pr, leakage)

    return nR, nS


def qos_A(gPP: torch.Tensor, p: CFParams) -> torch.Tensor:
    """QoS-derived interference budget A.

    Derived from primary constraint (paper-style):
      (1-tau) * 0.5 log2(1 + Pp*gPP / (1+Q)) >= Rp0
    which yields an equivalent bound Q <= A.

    Note: If the raw A is negative (primary can't meet target even at Q=0), we clamp
    A to 0 to keep the constraint set non-empty for learning/baselines.
    """
    tau = torch.tensor(p.tau, device=gPP.device, dtype=gPP.dtype)
    denom = torch.pow(
        torch.tensor(2.0, device=gPP.device, dtype=gPP.dtype),
        2.0 * p.Rp0 / (1.0 - tau),
    ) - 1.0
    A = p.Pp * gPP / denom - 1.0
    return torch.clamp(A, min=0.0)


def qos_Q_cf(g: Dict[str, torch.Tensor], ps: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
    """CF QoS metric is affine in powers (simpler than DF)."""
    gSP = _bcast_link(g["SP"], ps)
    gRP = _bcast_link(g["RP"], pr)
    return gSP * ps + gRP * pr


def K_terms(
    g: Dict[str, torch.Tensor],
    nR: torch.Tensor,
    nS: torch.Tensor,
    rho_z: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute K1 and K2 as defined in the paper."""
    rho = torch.tensor(rho_z, device=nR.device, dtype=nR.dtype)

    gSR = _bcast_link(g["SR"], nR)
    gSS = _bcast_link(g["SS"], nR)

    # K1 mixes the two noisy observations (relay and destination) and includes correlation term rho_z
    mix = safe_sqrt((gSR * gSS) / (nS * nR))
    K1 = gSR / nS + gSS / nR - 2.0 * rho * mix

    # K2 is the remaining noise coupling term (positive when |rho|<1)
    K2 = (1.0 - rho ** 2) / (nR * nS)
    return K1, K2


def secondary_rate_cf(
    g: Dict[str, torch.Tensor],
    ps: torch.Tensor,
    pr: torch.Tensor,
    p: CFParams,
    hw: HardwareConfig = HardwareConfig(),
) -> torch.Tensor:
    """Secondary achievable rate under CF as per the extracted expression."""
    nR, nS = effective_noises(g, p, pr, hw)
    K1, K2 = K_terms(g, nR, nS, p.rho_z)

    gRS = _bcast_link(g["RS"], pr)
    gSS = _bcast_link(g["SS"], ps)

    num = K1 * gRS * ps * pr + gSS * ps * (K1 * ps + K2)  # numerator from paper
    den = K2 * gRS * pr + nS * (K1 * ps + K2)             # denominator from paper

    snr_eff = num / den.clamp_min(1e-12)  # avoid division by zero
    return capacity_awgn(snr_eff)


def primary_rate_cf(
    g: Dict[str, torch.Tensor],
    ps: torch.Tensor,
    pr: torch.Tensor,
    p: CFParams,
) -> torch.Tensor:
    """Primary rate for evaluation under CF."""
    Q = qos_Q_cf(g, ps, pr)
    gPP = _bcast_link(g["PP"], Q)
    snr_p = p.Pp * gPP / (1.0 + Q)
    return (1.0 - p.tau) * capacity_awgn(snr_p)
