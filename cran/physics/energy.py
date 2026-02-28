# cran/physics/energy.py

"""Energy efficiency metrics."""

from __future__ import annotations

import torch

from cran.utils.math_utils import energy_efficiency


def ee_secondary(rs: torch.Tensor, ps: torch.Tensor, pr: torch.Tensor, pc: float = 0.5) -> torch.Tensor:
    """Energy efficiency of the secondary system."""
    return energy_efficiency(rs, ps, pr, p_c=pc)
