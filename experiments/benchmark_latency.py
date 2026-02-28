# experiments/benchmark_latency.py

"""Benchmark inference latency for a policy model (CPU/GPU).

Outputs:
- runs/benchmark_latency/latency.json
- optional latency figure (Phase9 deploy_plots.py)

This provides the 'real-time evidence' contribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.policies.df_policy import DFPolicy
from cran.deploy.benchmark_latency import benchmark_model, BenchmarkConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/deploy.yaml"])
    ap.add_argument("--out", default="runs/benchmark_latency")
    ap.add_argument("--in_dim", type=int, default=8)
    ap.add_argument("--model", choices=["df"], default="df")
    args = ap.parse_args()

    cfg = merge_configs(args.configs)
    device = get_torch_device(cfg["device"]["prefer"])
    set_global_seed(SeedConfig(value=cfg["seed"]["value"],
                               deterministic=cfg["project"]["deterministic"],
                               cudnn_benchmark=cfg["device"]["cudnn_benchmark"]))
    logger = build_logger("cran-ai", cfg["logging"]["level"], f"{args.out}/run.log" if cfg["logging"]["log_to_file"] else None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out / "config_merged.yaml")

    batch_sizes = cfg["deploy"]["benchmark"]["batch_sizes"]
    warm = cfg["deploy"]["benchmark"]["warmup_iters"]
    meas = cfg["deploy"]["benchmark"]["measure_iters"]

    if args.model == "df":
        model = DFPolicy(in_dim=args.in_dim)

    latency = benchmark_model(model, device, in_dim=args.in_dim, batch_sizes=batch_sizes,
                              cfg=BenchmarkConfig(warmup_iters=warm, measure_iters=meas))

    with open(out / "latency.json", "w", encoding="utf-8") as f:
        json.dump({"device": str(device), "latency_ms": latency}, f, indent=2)

    logger.info(f"saved latency.json on device={device}")

if __name__ == "__main__":
    main()
