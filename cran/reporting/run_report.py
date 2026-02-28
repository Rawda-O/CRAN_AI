from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from cran.reporting.artifact_registry import ArtifactPaths


def write_markdown_report(out_dir: str | Path,
                          title: str,
                          summary: Dict,
                          notes: Optional[str] = None) -> Path:
    ap = ArtifactPaths.make(out_dir)
    p = ap.reports / "technical_report.md"

    lines = []
    lines.append(f"# {title}\n")
    lines.append("## Summary\n")
    for k, v in summary.items():
        lines.append(f"- **{k}**: {v}\n")
    if notes:
        lines.append("\n## Notes\n")
        lines.append(notes + "\n")
    lines.append("\n## Artifacts\n")
    lines.append(f"- Results: `{ap.results}`\n")
    lines.append(f"- Figures: `{ap.figures}`\n")
    lines.append(f"- Tables: `{ap.tables}`\n")
    lines.append(f"- Checkpoints: `{ap.checkpoints}`\n")

    p.write_text("".join(lines), encoding="utf-8")
    return p
