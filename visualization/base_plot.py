# visualization/base_plot.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass(frozen=True)
class PlotConfig:
    save_dir: str = "figures"
    formats: tuple = ("png", "pdf")
    dpi: int = 200
    seaborn_theme: str = "whitegrid"  # matches config default


def setup_theme(cfg: PlotConfig) -> None:
    """Apply a consistent theme (paper-friendly)."""
    sns.set_theme(style=cfg.seaborn_theme)  # consistent aesthetics across figures


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_figure(fig: plt.Figure, out_dir: str | Path, name: str, cfg: PlotConfig) -> None:
    out = ensure_dir(out_dir)
    for fmt in cfg.formats:
        fig.savefig(out / f"{name}.{fmt}", dpi=cfg.dpi, bbox_inches="tight")  # archive-ready
