# cran/utils/logger.py

"""Logging utilities.

We keep logging minimal but structured:
- Console logging for real-time feedback
- Optional file logging for experiment archives
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def build_logger(name: str = "cran-ai",
                 level: str = "INFO",
                 log_file: Optional[str] = None) -> logging.Logger:
    """Create a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)  # console output

    if log_file is not None:
        p = Path(log_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)  # persistent archive

    logger.propagate = False
    return logger
