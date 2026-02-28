from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    results: Path
    figures: Path
    tables: Path
    checkpoints: Path
    reports: Path

    @staticmethod
    def make(root: str | Path) -> "ArtifactPaths":
        r = Path(root)
        p = ArtifactPaths(
            root=r,
            results=r / "results",
            figures=r / "figures",
            tables=r / "tables",
            checkpoints=r / "checkpoints",
            reports=r / "reports",
        )
        for d in [p.results, p.figures, p.tables, p.checkpoints, p.reports]:
            d.mkdir(parents=True, exist_ok=True)
        return p
