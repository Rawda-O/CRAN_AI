# cran/deploy/export_onnx.py

"""Export to ONNX (edge deployment)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch


def export_onnx(model: torch.nn.Module,
                example_input: torch.Tensor,
                out_path: str | Path,
                input_names: Sequence[str] = ("x",),
                output_names: Sequence[str] = ("y",),
                opset: int = 17) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        example_input,
        str(out),
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        do_constant_folding=True,
    )
    return out
