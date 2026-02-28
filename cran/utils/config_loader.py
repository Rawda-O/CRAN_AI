# cran/utils/config_loader.py

"""Config loading and merging.

Design goals:
- Use YAML configs for all experiment knobs.
- Merge multiple YAML files into a single resolved config dict.
- Keep the merged snapshot to guarantee reproducibility.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import yaml


def _deep_update(base: Dict[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dict.
    Values in `other` override values in `base`.
    """
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_update(dict(base[k]), v)   # recurse into nested dicts
        else:
            base[k] = copy.deepcopy(v)                # overwrite leaf value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file into a dict."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_configs(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Merge multiple YAML configs in order."""
    merged: Dict[str, Any] = {}
    for p in paths:
        merged = _deep_update(merged, load_yaml(p))   # later configs override earlier ones
    return merged


def save_config_snapshot(cfg: Mapping[str, Any], out_path: str | Path) -> None:
    """Save merged config for reproducibility."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)  # preserve human readability
