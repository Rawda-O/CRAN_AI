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

from visualization.multiobjective_plots import plot_tradeoff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/cf.yaml","configs/visualization.yaml"])
    ap.add_argument("--out", default="runs/sweep_channel_models")
    ap.add_argument("--models", nargs="+", default=["gaussian","anne","uniform","rician","nakagami"])
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
    dfp = DFParams(tau=df_cfg["tau"], noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])
    cfp = CFParams(tau=cf_cfg["tau"], noise_power=cf_cfg["noise_power"], Pp=cf_cfg["power"]["Pp"], Rp0=cf_cfg["primary_qos"]["Rp0"])

    rs_df_list, rs_cf_list = [], []

    for m in args.models:
        g = generate_gains(batch, m, device, geo=geo, fad=fad)
        df_sol = solve_df_bruteforce(g, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp, DFGrid(ps_steps=21, pr_steps=21, alpha_steps=11))
        cf_sol = solve_cf_bruteforce(g, cf_cfg["power"]["Ps_max"], cf_cfg["power"]["Pr_max"], cfp, CFGrid(ps_steps=41, pr_steps=41))
        rs_cf = secondary_rate_cf(g, cf_sol["ps"], cf_sol["pr"], cfp)
        rs_df = float(df_sol["rs"].mean().item())
        rs_cf_m = float(rs_cf.mean().item())
        rs_df_list.append(rs_df)
        rs_cf_list.append(rs_cf_m)
        logger.info(f"model={m} RS_DF={rs_df:.4f} RS_CF={rs_cf_m:.4f}")

    with open(out / "results" / "sweep_channel_models.json", "w", encoding="utf-8") as f:
        json.dump({"models": args.models, "rs_df": rs_df_list, "rs_cf": rs_cf_list}, f, indent=2)

    # Simple plot as a tradeoff-style line (index -> RS)
    xs = list(range(len(args.models)))
    plot_tradeoff(xs, rs_df_list, "Channel model index", "Mean R_S (DF)", out / "figures", "rs_df_vs_model", title="DF baseline across channel models")
    plot_tradeoff(xs, rs_cf_list, "Channel model index", "Mean R_S (CF)", out / "figures", "rs_cf_vs_model", title="CF baseline across channel models")

if __name__ == "__main__":
    main()
