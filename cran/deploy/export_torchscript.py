# cran/deploy/export_torchscript.py

"""Export a policy to TorchScript (deployable)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def export_torchscript(model: torch.nn.Module, example_input: torch.Tensor, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    scripted = torch.jit.trace(model, example_input)  # trace is sufficient for MLPs (simple, stable)
    scripted.save(str(out))
    return out
