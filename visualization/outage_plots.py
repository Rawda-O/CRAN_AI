from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt

from visualization.base_plot import PlotConfig, setup_theme, save_figure


def plot_outage_vs_snr(snr_db: Sequence[float],
                       curves: Dict[str, Sequence[float]],
                       out_dir: str | Path,
                       cfg: PlotConfig = PlotConfig(),
                       title: str = "Outage vs SNR") -> None:
    setup_theme(cfg)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for name, y in curves.items():
        ax.plot(snr_db, y, marker="o", label=name)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Outage Probability")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both")
    ax.legend()
    save_figure(fig, out_dir, "outage_vs_snr", cfg)
    plt.close(fig)
