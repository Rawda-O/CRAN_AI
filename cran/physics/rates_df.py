# cran/physics/rates_df.py

"""Decode-and-Forward (DF) achievable rates and QoS constraint.

Implemented directly from the DF expressions extracted from the attached paper text:
- RS = C( min{ fR, fS } )
- QoS constraint: Q = gSP*PS + gRP*PR + 2*alpha*sqrt(gSP*gRP*PS*PR)

New contributions enabled here:
- Works seamlessly with imperfect CSI pairing (input g_hat vs loss g).
- Compatible with Safety layer (post-processing to guarantee QoS).
- Compatible with hardware impairments via effective noise terms.

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
class DFParams:
    tau: float = 0.25
    noise_power: float = 1.0
    Pp: float = 10.0
    Rp0: float = 1.0


def effective_noises(
    g: Dict[str, torch.Tensor],
    p: DFParams,
    pr: torch.Tensor,
    hw: HardwareConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute \tilde{N}_R and \tilde{N}_S used in DF/CF formulas.

    \tilde{N}_R = N0 + Pp*g_PR (+ residual SI term if enabled)
    \tilde{N}_S = N0 + Pp*g_PS
    """
    n0 = torch.tensor(p.noise_power, device=pr.device, dtype=pr.dtype)  # base thermal noise (normalized)

    gPR = _bcast_link(g["PR"], pr)
    gPS = _bcast_link(g["PS"], pr)

    nR = n0 + p.Pp * gPR  # primary->relay interference term (affects relay decoding)
    nS = n0 + p.Pp * gPS  # primary->secondary interference term (affects secondary decoding)

    if hw.enable:
        leakage = si_linear(hw.residual_si_db)      # new contribution knob: residual SI level
        nR = apply_residual_si(nR, pr, leakage)     # SI increases relay effective noise

    return nR, nS


def qos_A(gPP: torch.Tensor, p: DFParams) -> torch.Tensor:
    """QoS-derived interference budget A.

    Same definition as CF for consistency.

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


def qos_Q_df(g: Dict[str, torch.Tensor], ps: torch.Tensor, pr: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """DF QoS expression Q(h,alpha,Ps,Pr) from the paper."""
    gSP = _bcast_link(g["SP"], ps)
    gRP = _bcast_link(g["RP"], pr)
    cross = 2.0 * alpha * safe_sqrt(gSP * gRP * ps * pr)  # nonconvex cross-term (DF-specific)
    return gSP * ps + gRP * pr + cross                    # total interference metric


def fR(g: Dict[str, torch.Tensor], ps: torch.Tensor, alpha: torch.Tensor, nR: torch.Tensor) -> torch.Tensor:
    """Relay decoding SNR term under DF: fR = gSR*(1-alpha^2)*Ps / \tilde{N}_R."""
    gSR = _bcast_link(g["SR"], ps)
    return gSR * (1.0 - alpha ** 2) * ps / nR


def fS(
    g: Dict[str, torch.Tensor],
    ps: torch.Tensor,
    pr: torch.Tensor,
    alpha: torch.Tensor,
    nS: torch.Tensor,
) -> torch.Tensor:
    """Destination combining SNR term under DF."""
    gSS = _bcast_link(g["SS"], ps)
    gRS = _bcast_link(g["RS"], pr)

    num = gSS * ps + gRS * pr + 2.0 * alpha * safe_sqrt(gRS * gSS * ps * pr)
    return num / nS


def secondary_rate_df(
    g: Dict[str, torch.Tensor],
    ps: torch.Tensor,
    pr: torch.Tensor,
    alpha: torch.Tensor,
    p: DFParams,
    hw: HardwareConfig = HardwareConfig(),
) -> torch.Tensor:
    """Secondary achievable rate under DF: RS = C(min{fR,fS})."""
    nR, nS = effective_noises(g, p, pr, hw)
    snr = torch.minimum(fR(g, ps, alpha, nR), fS(g, ps, pr, alpha, nS))
    return capacity_awgn(snr)


def primary_rate_df(
    g: Dict[str, torch.Tensor],
    ps: torch.Tensor,
    pr: torch.Tensor,
    alpha: torch.Tensor,
    p: DFParams,
) -> torch.Tensor:
    """Primary rate expression used for evaluation (not only the constraint)."""
    # Primary sees interference metric Q; its effective SNR is Pp*gPP / (1 + Q)
    Q = qos_Q_df(g, ps, pr, alpha)
    gPP = _bcast_link(g["PP"], Q)
    snr_p = p.Pp * gPP / (1.0 + Q)
    return (1.0 - p.tau) * capacity_awgn(snr_p)  # weighted by (1-tau) time fraction
