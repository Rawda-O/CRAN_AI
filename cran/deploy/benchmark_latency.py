# cran/deploy/benchmark_latency.py

"""Latency benchmark (CPU/GPU).

Contribution focus:
- Produce 'real-time evidence' by reporting latency vs batch size.
- Uses torch.cuda.synchronize() to correctly measure GPU time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch


@dataclass(frozen=True)
class BenchmarkConfig:
    warmup_iters: int = 50
    measure_iters: int = 200


@torch.no_grad()
def benchmark_model(model: torch.nn.Module,
                    device: torch.device,
                    in_dim: int,
                    batch_sizes: Sequence[int],
                    cfg: BenchmarkConfig = BenchmarkConfig()) -> Dict[int, float]:
    model = model.to(device).eval()

    results_ms: Dict[int, float] = {}
    for bs in batch_sizes:
        x = torch.randn(bs, in_dim, device=device)

        # warmup (important on GPU: triggers kernel compilation/caching)
        for _ in range(cfg.warmup_iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # measure
        t0 = time.perf_counter()
        for _ in range(cfg.measure_iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_ms = (t1 - t0) * 1000.0 / cfg.measure_iters
        results_ms[int(bs)] = float(avg_ms)

    return results_ms
