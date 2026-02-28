from __future__ import annotations

import argparse
from pathlib import Path
import json

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig
from cran.physics.csi import ImperfectCSIConfig, make_imperfect_pair
from cran.physics.rates_df import DFParams
from cran.physics.rates_cf import CFParams
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid
from cran.baselines.bruteforce_cf import solve_cf_bruteforce, CFGrid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/cf.yaml","configs/robust.yaml"])
    ap.add_argument("--out", default="runs/smoke_baselines")
    args = ap.parse_args()

    cfg = merge_configs(args.configs)
    device = get_torch_device(cfg["device"]["prefer"])
    set_global_seed(SeedConfig(value=cfg["seed"]["value"],
                               deterministic=cfg["project"]["deterministic"],
                               cudnn_benchmark=cfg["device"]["cudnn_benchmark"]))
    logger = build_logger("cran-ai", cfg["logging"]["level"], f"{args.out}/run.log" if cfg["logging"]["log_to_file"] else None)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    save_config_snapshot(cfg, f"{args.out}/config_merged.yaml")

    batch = 256
    ch = cfg["channel"]
    geo = GeometryConfig(**ch["geometry"])
    fad = FadingConfig(**ch["fading"])
    g = generate_gains(batch, ch["model"], device, geo=geo, fad=fad, mixed_weights=ch.get("mixed",{}).get("weights"))

    if cfg["robust"]["enable"]:
        icfg = ImperfectCSIConfig(sigma=cfg["robust"]["noise"]["sigma"],
                                  apply_to_links=tuple(cfg["robust"]["noise"]["apply_to_links"]),
                                  clamp_min=cfg["robust"]["noise"]["clamp_min"])
        _, g_ref = make_imperfect_pair(g, icfg)
    else:
        g_ref = g

    df_cfg = cfg["df"]
    cf_cfg = cfg["cf"]
    dfp = DFParams(tau=df_cfg["tau"], noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])
    cfp = CFParams(tau=cf_cfg["tau"], noise_power=cf_cfg["noise_power"], Pp=cf_cfg["power"]["Pp"], Rp0=cf_cfg["primary_qos"]["Rp0"])

    df_sol = solve_df_bruteforce(g_ref, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp, DFGrid(ps_steps=21, pr_steps=21, alpha_steps=11))
    cf_sol = solve_cf_bruteforce(g_ref, cf_cfg["power"]["Ps_max"], cf_cfg["power"]["Pr_max"], cfp, CFGrid(ps_steps=41, pr_steps=41))

    logger.info(f"DF brute-force mean RS: {df_sol['rs'].mean().item():.4f}")
    logger.info(f"CF brute-force mean RS: {cf_sol['rs'].mean().item():.4f}")

    with open(f"{args.out}/summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "device": str(device),
            "df_mean_rs": float(df_sol["rs"].mean().item()),
            "cf_mean_rs": float(cf_sol["rs"].mean().item()),
        }, f, indent=2)

if __name__ == "__main__":
    main()
