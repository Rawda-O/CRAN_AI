from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Sequence

import torch


@dataclass(frozen=True)
class ThroughputConfig:
    warmup_iters: int = 50
    measure_iters: int = 200


@torch.no_grad()
def benchmark_throughput(model: torch.nn.Module,
                         device: torch.device,
                         in_dim: int,
                         batch_sizes: Sequence[int],
                         cfg: ThroughputConfig = ThroughputConfig()) -> Dict[int, float]:
    model = model.to(device).eval()
    out: Dict[int, float] = {}

    for bs in batch_sizes:
        x = torch.randn(bs, in_dim, device=device)

        for _ in range(cfg.warmup_iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(cfg.measure_iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_samples = bs * cfg.measure_iters
        sps = total_samples / (t1 - t0)
        out[int(bs)] = float(sps)

    return out
