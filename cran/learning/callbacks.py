# cran/learning/callbacks.py

"""Callbacks (minimal).

We keep callbacks lean to avoid hidden behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStopper:
    patience: int = 10
    best: float = float("inf")
    bad_epochs: int = 0
    mode: str = "min"  # min loss by default

    def step(self, value: float) -> bool:
        improved = (value < self.best) if self.mode == "min" else (value > self.best)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
