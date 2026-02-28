from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig
from cran.physics.rates_df import DFParams
from cran.physics.rates_cf import CFParams, secondary_rate_cf
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid
from cran.baselines.bruteforce_cf import solve_cf_bruteforce, CFGrid

from visualization.generalization_plots import plot_rs_vs_tau


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/cf.yaml","configs/visualization.yaml"])
    ap.add_argument("--out", default="runs/sweep_power_tau")
    ap.add_argument("--tau", nargs="+", type=float, default=[0.2,0.3,0.4,0.5,0.6,0.7,0.8])
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

    df_cfg = cfg["df"]
    cf_cfg = cfg["cf"]

    curves = {"DF(bruteforce)": [], "CF(bruteforce)": []}

    for tau in args.tau:
        g = generate_gains(batch, ch["model"], device, geo=geo, fad=fad, mixed_weights=ch.get("mixed",{}).get("weights"))
        dfp = DFParams(tau=float(tau), noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])
        cfp = CFParams(tau=float(tau), noise_power=cf_cfg["noise_power"], Pp=cf_cfg["power"]["Pp"], Rp0=cf_cfg["primary_qos"]["Rp0"])

        df_sol = solve_df_bruteforce(g, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp, DFGrid(ps_steps=21, pr_steps=21, alpha_steps=11))
        cf_sol = solve_cf_bruteforce(g, cf_cfg["power"]["Ps_max"], cf_cfg["power"]["Pr_max"], cfp, CFGrid(ps_steps=41, pr_steps=41))
        rs_cf = secondary_rate_cf(g, cf_sol["ps"], cf_sol["pr"], cfp)

        curves["DF(bruteforce)"].append(float(df_sol["rs"].mean().item()))
        curves["CF(bruteforce)"].append(float(rs_cf.mean().item()))
        logger.info(f"tau={tau:.2f} RS_DF={curves['DF(bruteforce)'][-1]:.4f} RS_CF={curves['CF(bruteforce)'][-1]:.4f}")

    with open(out / "results" / "sweep_tau.json", "w", encoding="utf-8") as f:
        json.dump({"tau": list(map(float,args.tau)), "rs": curves}, f, indent=2)

    plot_rs_vs_tau(args.tau, curves, out / "figures")

if __name__ == "__main__":
    main()
