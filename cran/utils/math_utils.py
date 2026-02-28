# cran/utils/math_utils.py

"""Mathematical utility layer for the 6G-ready AI-native Cognitive Relay Framework.

Design principles:
- Pure PyTorch
- Fully GPU compatible
- Fully batch compatible
- Numerically stable
- No communication-specific logic here
"""

import torch


# ============================================================
# Core Mathematical Operations
# ============================================================

def log2(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Numerically stable base-2 logarithm."""
    x = torch.clamp(x, min=eps)  # prevent log(0) instability
    return torch.log(x) / torch.log(torch.tensor(2.0, device=x.device))  # convert ln to log2


def capacity_awgn(snr: torch.Tensor) -> torch.Tensor:
    """Shannon capacity for AWGN channel: C = 0.5 * log2(1 + SNR)."""
    return 0.5 * log2(1.0 + snr)  # 0.5 due to half-duplex equivalent relay timing


def relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU wrapper used for soft-constraint penalties."""
    return torch.relu(x)  # ensures non-negative violation term


def clamp_unit_interval(x: torch.Tensor) -> torch.Tensor:
    """Clamp values to [0, 1] (used for gamma power fractions)."""
    return torch.clamp(x, 0.0, 1.0)  # enforce physical bounds


def safe_sqrt(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Safe sqrt (protects against tiny negative float noise)."""
    return torch.sqrt(torch.clamp(x, min=0.0))


# ============================================================
# Energy Efficiency
# ============================================================

def energy_efficiency(rate: torch.Tensor,
                      p_s: torch.Tensor,
                      p_r: torch.Tensor,
                      p_c: float = 0.5) -> torch.Tensor:
    """EE = R_s / (P_s + P_r + P_c)."""
    total_power = p_s + p_r + p_c  # total consumed power including circuit power
    return rate / torch.clamp(total_power, min=1e-12)  # avoid division by zero


# ============================================================
# Outage Utilities
# ============================================================

def outage_indicator(rate: torch.Tensor, threshold: float) -> torch.Tensor:
    """Hard outage: 1 if rate < threshold else 0 (for evaluation)."""
    return (rate < threshold).float()  # boolean -> float for averaging


def smooth_outage_proxy(rate: torch.Tensor, threshold: float, k: float = 10.0) -> torch.Tensor:
    """Differentiable outage proxy using sigmoid (for training)."""
    return torch.sigmoid(k * (threshold - rate))  # smooth approximation of outage event
