# cran/selection/selector_net.py

"""DNN-based selector.

Input design (contribution-critical):
- gains (imperfect or perfect) +
- optional DF/CF predicted gammas (include_policy_outputs=True)
Output:
- probability p(DF) in [0,1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


class SelectorNet(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...] = (128, 128), dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits)  # p(DF)
