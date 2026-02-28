from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_bundle(path: str | Path, **items) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, p)
    return p


def load_bundle(path: str | Path, map_location: Optional[str] = None) -> Dict[str, Any]:
    return torch.load(str(path), map_location=map_location)
