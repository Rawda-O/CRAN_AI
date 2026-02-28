from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig
from cran.physics.csi import ImperfectCSIConfig, make_imperfect_pair
from cran.physics.rates_df import DFParams, qos_A, qos_Q_df
from cran.physics.rates_cf import CFParams, qos_Q_cf, qos_A as qos_A_cf, secondary_rate_cf
from cran.physics.outage import outage_hard
from cran.physics.energy import ee_secondary
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid
from cran.baselines.bruteforce_cf import solve_cf_bruteforce, CFGrid

from visualization.rate_plots import plot_rs_vs_snr
from visualization.outage_plots import plot_outage_vs_snr
from visualization.qos_plots import plot_qos_violation_vs_snr
from visualization.energy_plots import plot_ee_vs_snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/cf.yaml","configs/robust.yaml","configs/multi_objective.yaml","configs/visualization.yaml"])
    ap.add_argument("--out", default="runs/sweep_snr")
    ap.add_argument("--snr_db", nargs="+", type=float, default=[0, 5, 10, 15, 20, 25, 30])
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

    rs_th = cfg["multi_objective"]["outage"]["Rs_threshold"]

    curves_rs = {"DF(bruteforce)": [], "CF(bruteforce)": []}
    curves_out = {"DF(bruteforce)": [], "CF(bruteforce)": []}
    curves_qos = {"DF(bruteforce)": [], "CF(bruteforce)": []}
    curves_ee = {"DF(bruteforce)": [], "CF(bruteforce)": []}

    for snr_db in args.snr_db:
        g = generate_gains(batch, ch["model"], device, geo=geo, fad=fad, mixed_weights=ch.get("mixed",{}).get("weights"))
        scale = 10.0 ** (snr_db / 10.0)
        g = {k: v * scale for k, v in g.items()}

        if cfg["robust"]["enable"]:
            icfg = ImperfectCSIConfig(sigma=cfg["robust"]["noise"]["sigma"],
                                      apply_to_links=tuple(cfg["robust"]["noise"]["apply_to_links"]),
                                      clamp_min=cfg["robust"]["noise"]["clamp_min"])
            _, g_ref = make_imperfect_pair(g, icfg)
        else:
            g_ref = g

        df_sol = solve_df_bruteforce(g_ref, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp,
                                     DFGrid(ps_steps=21, pr_steps=21, alpha_steps=11))
        cf_sol = solve_cf_bruteforce(g_ref, cf_cfg["power"]["Ps_max"], cf_cfg["power"]["Pr_max"], cfp,
                                     CFGrid(ps_steps=41, pr_steps=41))

        A_df = qos_A(g_ref["PP"], dfp)
        qos_v_df = torch.relu(qos_Q_df(g_ref, df_sol["ps"], df_sol["pr"], df_sol["alpha"]) - A_df).mean().item()

        A_cf = qos_A_cf(g_ref["PP"], cfp)
        rs_cf = secondary_rate_cf(g_ref, cf_sol["ps"], cf_sol["pr"], cfp)
        qos_v_cf = torch.relu(qos_Q_cf(g_ref, cf_sol["ps"], cf_sol["pr"]) - A_cf).mean().item()

        out_df = outage_hard(df_sol["rs"], rs_th).mean().item()
        out_cf = outage_hard(rs_cf, rs_th).mean().item()

        ee_df = ee_secondary(df_sol["rs"], df_sol["ps"], df_sol["pr"]).mean().item()
        ee_cf = ee_secondary(rs_cf, cf_sol["ps"], cf_sol["pr"]).mean().item()

        curves_rs["DF(bruteforce)"].append(float(df_sol["rs"].mean().item()))
        curves_rs["CF(bruteforce)"].append(float(rs_cf.mean().item()))
        curves_out["DF(bruteforce)"].append(float(out_df))
        curves_out["CF(bruteforce)"].append(float(out_cf))
        curves_qos["DF(bruteforce)"].append(float(qos_v_df))
        curves_qos["CF(bruteforce)"].append(float(qos_v_cf))
        curves_ee["DF(bruteforce)"].append(float(ee_df))
        curves_ee["CF(bruteforce)"].append(float(ee_cf))

        logger.info(f"SNR={snr_db:>4.1f} dB | RS_DF={curves_rs['DF(bruteforce)'][-1]:.4f} | RS_CF={curves_rs['CF(bruteforce)'][-1]:.4f}")

    results = {"snr_db": list(map(float, args.snr_db)), "rs": curves_rs, "outage": curves_out, "qos_violation": curves_qos, "ee": curves_ee}
    with open(out / "results" / "sweep_snr.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_rs_vs_snr(args.snr_db, curves_rs, out / "figures")
    plot_outage_vs_snr(args.snr_db, curves_out, out / "figures")
    plot_qos_violation_vs_snr(args.snr_db, curves_qos, out / "figures")
    plot_ee_vs_snr(args.snr_db, curves_ee, out / "figures")


if __name__ == "__main__":
    main()
