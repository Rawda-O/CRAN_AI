# cran/physics/channels.py

"""Channel coefficient/gain generation.

We always output **channel gains** g_ij >= 0 for the 8 fixed links:
PP, PR, RP, SS, SR, RS, SP, PS.

Supported channel models (as in Extended2 notebooks):
- gaussian: geometry + pathloss + Gaussian fading  (paper-consistent baseline)
- anne:     deterministic pathloss-only model      ("Anne" in notebook naming)
- uniform:  i.i.d. uniform gains (link-only model)
- rician:   i.i.d. Rician gains  (link-only model)
- nakagami: i.i.d. Nakagami gains(link-only model)
- mixed:    domain randomization across models (new contribution: cross-model generalization)

GPU-first design:
- pure torch ops (no numpy/scipy) so generation can run on GPU for large batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch


LINKS: Tuple[str, ...] = ("PP","PR","RP","SS","SR","RS","SP","PS")


@dataclass(frozen=True)
class GeometryConfig:
    cell_size: float = 10.0            # square cell side length
    relay_xy: Tuple[float, float] = (5.0, 5.0)  # relay location (fixed)
    pathloss_exp: float = 3.0          # gamma in papers
    sigma_g: float = 7.0              # Gaussian fading std in papers
    min_distance: float = 1e-2        # avoid division by zero


@dataclass(frozen=True)
class FadingConfig:
    rician_b: float = 0.775
    nakagami_nu: float = 4.97
    uniform_low: float = 1e-3
    uniform_high: float = 1e3


def _sample_xy(batch: int, cell_size: float, device: torch.device) -> torch.Tensor:
    """Sample (x,y) uniformly in a square cell."""
    return torch.rand(batch, 2, device=device) * cell_size  # uniform spatial distribution


def _pairwise_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Euclidean distance between two (batch,2) tensors."""
    return torch.sqrt(torch.sum((a - b) ** 2, dim=-1) + eps)  # eps prevents sqrt(0)


def _geometry_nodes(batch: int, geo: GeometryConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Generate node locations.

    Paper-consistent approach: all terminals are randomly placed within the cell,
    while the relay is fixed at relay_xy (common assumption in the provided papers).
    """
    up = _sample_xy(batch, geo.cell_size, device)              # primary transmitter location
    dp = _sample_xy(batch, geo.cell_size, device)              # primary receiver location
    us = _sample_xy(batch, geo.cell_size, device)              # secondary transmitter location
    ds = _sample_xy(batch, geo.cell_size, device)              # secondary receiver location
    r = torch.tensor(geo.relay_xy, device=device).view(1,2).repeat(batch,1)  # fixed relay location
    return {"UP": up, "DP": dp, "US": us, "DS": ds, "R": r}


def _pathloss(d: torch.Tensor, gamma: float) -> torch.Tensor:
    """Pathloss denominator: sqrt(1 + d^gamma)."""
    return torch.sqrt(1.0 + d ** gamma)  # matches paper form 1/sqrt(1+d^gamma)


def _gaussian_geometry_gains(batch: int, geo: GeometryConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Model (1): geometry + Gaussian fading, consistent with the papers."""
    nodes = _geometry_nodes(batch, geo, device)
    # Distances for the 8 links
    d_PP = _pairwise_distance(nodes["UP"], nodes["DP"])
    d_PR = _pairwise_distance(nodes["UP"], nodes["R"])
    d_RP = _pairwise_distance(nodes["R"], nodes["DP"])
    d_SS = _pairwise_distance(nodes["US"], nodes["DS"])
    d_SR = _pairwise_distance(nodes["US"], nodes["R"])
    d_RS = _pairwise_distance(nodes["R"], nodes["DS"])
    d_SP = _pairwise_distance(nodes["US"], nodes["DP"])
    d_PS = _pairwise_distance(nodes["UP"], nodes["DS"])

    # Gaussian fading coefficient s ~ N(0, sigma_g^2)
    s = torch.randn(batch, len(LINKS), device=device) * geo.sigma_g  # per-link independent fading samples
    # Apply geometry pathloss: h = s / sqrt(1 + d^gamma)
    denom = torch.stack([
        _pathloss(d_PP, geo.pathloss_exp),
        _pathloss(d_PR, geo.pathloss_exp),
        _pathloss(d_RP, geo.pathloss_exp),
        _pathloss(d_SS, geo.pathloss_exp),
        _pathloss(d_SR, geo.pathloss_exp),
        _pathloss(d_RS, geo.pathloss_exp),
        _pathloss(d_SP, geo.pathloss_exp),
        _pathloss(d_PS, geo.pathloss_exp),
    ], dim=-1).clamp_min(geo.min_distance)  # safety clamp for numerical stability

    h = s / denom  # coefficient
    g = h ** 2     # gain (matches notebook conversion g = h^2)

    return {k: g[:, i] for i, k in enumerate(LINKS)}


def _anne_pathloss_only_gains(batch: int, geo: GeometryConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Model (2): "Anne" model in notebook naming: deterministic pathloss-only gains.
    h = 1 / d^(3/2)  =>  g = 1 / d^3
    """
    nodes = _geometry_nodes(batch, geo, device)
    d_PP = _pairwise_distance(nodes["UP"], nodes["DP"]).clamp_min(geo.min_distance)
    d_PR = _pairwise_distance(nodes["UP"], nodes["R"]).clamp_min(geo.min_distance)
    d_RP = _pairwise_distance(nodes["R"], nodes["DP"]).clamp_min(geo.min_distance)
    d_SS = _pairwise_distance(nodes["US"], nodes["DS"]).clamp_min(geo.min_distance)
    d_SR = _pairwise_distance(nodes["US"], nodes["R"]).clamp_min(geo.min_distance)
    d_RS = _pairwise_distance(nodes["R"], nodes["DS"]).clamp_min(geo.min_distance)
    d_SP = _pairwise_distance(nodes["US"], nodes["DP"]).clamp_min(geo.min_distance)
    d_PS = _pairwise_distance(nodes["UP"], nodes["DS"]).clamp_min(geo.min_distance)

    g = torch.stack([
        d_PP.pow(-3.0),
        d_PR.pow(-3.0),
        d_RP.pow(-3.0),
        d_SS.pow(-3.0),
        d_SR.pow(-3.0),
        d_RS.pow(-3.0),
        d_SP.pow(-3.0),
        d_PS.pow(-3.0),
    ], dim=-1)  # directly gains (no random fading)

    return {k: g[:, i] for i, k in enumerate(LINKS)}


def _uniform_link_gains(batch: int, fad: FadingConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Model (3): i.i.d. uniform gains (link-only, no geometry)."""
    g = torch.empty(batch, len(LINKS), device=device).uniform_(fad.uniform_low, fad.uniform_high)
    return {k: g[:, i] for i, k in enumerate(LINKS)}


def _rician_link_gains(batch: int, fad: FadingConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Model (4): i.i.d. Rician gains (link-only).

    Torch-native Rician magnitude sampling:
    If X,Y ~ N(0, sigma^2) and nu is LOS component,
    then r = sqrt((X+nu)^2 + Y^2) is Rician.
    We set sigma=1 and nu=b (so b = nu/sigma).
    """
    sigma = 1.0
    nu = fad.rician_b  # b = nu/sigma, sigma=1
    x = torch.randn(batch, len(LINKS), device=device) * sigma
    y = torch.randn(batch, len(LINKS), device=device) * sigma
    r = torch.sqrt((x + nu) ** 2 + y ** 2)  # Rician magnitude
    g = r ** 2  # gain
    return {k: g[:, i] for i, k in enumerate(LINKS)}


def _nakagami_link_gains(batch: int, fad: FadingConfig, device: torch.device) -> Dict[str, torch.Tensor]:
    """Model (5): i.i.d. Nakagami-m gains (link-only).

    If power ~ Gamma(m, scale=omega/m), then amplitude = sqrt(power) is Nakagami.
    We use omega=1 for normalization (can be changed later if needed).
    """
    m = torch.tensor(fad.nakagami_nu, device=device)
    omega = torch.tensor(1.0, device=device)
    # Gamma distribution for power
    shape = m
    scale = omega / m
    power = torch.distributions.Gamma(shape, 1.0 / scale).sample((batch, len(LINKS))).to(device)  # rate=1/scale
    g = power  # power itself is gain (since amplitude^2)
    return {k: g[:, i] for i, k in enumerate(LINKS)}


def generate_gains(batch: int,
                   model: str,
                   device: torch.device,
                   geo: Optional[GeometryConfig] = None,
                   fad: Optional[FadingConfig] = None,
                   mixed_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
    """Main factory: generate gains for the selected channel model."""
    geo = geo or GeometryConfig()
    fad = fad or FadingConfig()

    model = model.lower()
    if model == "gaussian":
        return _gaussian_geometry_gains(batch, geo, device)
    if model == "anne":
        return _anne_pathloss_only_gains(batch, geo, device)
    if model == "uniform":
        return _uniform_link_gains(batch, fad, device)
    if model == "rician":
        return _rician_link_gains(batch, fad, device)
    if model == "nakagami":
        return _nakagami_link_gains(batch, fad, device)

    # New contribution: mixed channel training (domain randomization) to improve generalization.
    if model == "mixed":
        if not mixed_weights:
            raise ValueError("mixed model requires mixed_weights dict")
        # normalize weights
        keys = list(mixed_weights.keys())
        w = torch.tensor([mixed_weights[k] for k in keys], device=device, dtype=torch.float32)
        w = w / w.sum().clamp_min(1e-12)
        # sample model index per sample
        idx = torch.multinomial(w, num_samples=batch, replacement=True)
        # generate per-model gains in chunks
        out = {k: torch.empty(batch, device=device) for k in LINKS}
        for mi, mk in enumerate(keys):
            mask = (idx == mi)
            if not torch.any(mask):
                continue
            g_chunk = generate_gains(int(mask.sum().item()), mk, device, geo=geo, fad=fad)
            for lk in LINKS:
                out[lk][mask] = g_chunk[lk]
        return out

    raise ValueError(f"Unknown channel model: {model}")
