from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig
from cran.physics.rates_df import DFParams, secondary_rate_df
from cran.physics.impairments import HardwareConfig
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid

from visualization.hardware_plots import plot_rs_vs_si_db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/visualization.yaml"])
    ap.add_argument("--out", default="runs/sweep_hardware_si")
    ap.add_argument("--si_db", nargs="+", type=float, default=[float("inf"), 80, 70, 60, 50, 40])
    args = ap.parse_args()

    cfg = merge_configs(args.configs)
    device = get_torch_device(cfg["device"]["prefer"])
    set_global_seed(SeedConfig(value=cfg["seed"]["value"],
                               deterministic=cfg["project"]["deterministic"],
                               cudnn_benchmark=cfg["device"]["cudnn_benchmark"]))
    logger = build_logger("cran-ai", cfg["logging"]["level"], f"{args.out}/run.log" if cfg["logging"]["log_to_file"] else None)

    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, out / "config_merged.yaml")

    batch = 512
    ch = cfg["channel"]
    geo = GeometryConfig(**ch["geometry"])
    fad = FadingConfig(**ch["fading"])
    g = generate_gains(batch, ch["model"], device, geo=geo, fad=fad)

    df_cfg = cfg["df"]
    dfp = DFParams(tau=df_cfg["tau"], noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])

    # Use baseline action, then evaluate rate under SI impairment
    sol = solve_df_bruteforce(g, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp, DFGrid(ps_steps=21, pr_steps=21, alpha_steps=11))
    curves = {"DF(bruteforce)": []}

    for si in args.si_db:
        hw = HardwareConfig(enable=True, residual_si_db=float(si))
        rs = secondary_rate_df(g, sol["ps"], sol["pr"], sol["alpha"], dfp, hw=hw)
        curves["DF(bruteforce)"].append(float(rs.mean().item()))
        logger.info(f"SI={si} dB RS={curves['DF(bruteforce)'][-1]:.4f}")

    with open(out / "results" / "sweep_si.json", "w", encoding="utf-8") as f:
        json.dump({"si_db": list(map(float,args.si_db)), "rs": curves}, f, indent=2)

    plot_rs_vs_si_db(args.si_db, curves, out / "figures")

if __name__ == "__main__":
    main()
