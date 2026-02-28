from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.policies.df_policy import DFPolicy
from cran.policies.cf_policy import CFPolicy
from cran.selection.rule_based import select_best_scheme
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid
from cran.baselines.bruteforce_cf import solve_cf_bruteforce, CFGrid

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig, LINKS
from cran.physics.rates_df import DFParams
from cran.physics.rates_cf import CFParams, secondary_rate_cf
from cran.deploy.device_report import build_device_report
from cran.reporting.artifact_registry import ArtifactPaths
from visualization.rate_plots import plot_rs_vs_snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/cf.yaml","configs/visualization.yaml"])
    ap.add_argument("--out", default="runs/evaluate_full_system")
    ap.add_argument("--snr_db", nargs="+", type=float, default=[0,5,10,15,20,25,30])
    ap.add_argument("--df_ckpt", default=None)
    ap.add_argument("--cf_ckpt", default=None)
    args = ap.parse_args()

    cfg = merge_configs(args.configs)
    device = get_torch_device(cfg["device"]["prefer"])
    set_global_seed(SeedConfig(value=cfg["seed"]["value"],
                               deterministic=cfg["project"]["deterministic"],
                               cudnn_benchmark=cfg["device"]["cudnn_benchmark"]))
    logger = build_logger("cran-ai", cfg["logging"]["level"], f"{args.out}/run.log" if cfg["logging"]["log_to_file"] else None)

    apaths = ArtifactPaths.make(args.out)
    save_config_snapshot(cfg, apaths.root / "config_merged.yaml")
    (apaths.results / "device_report.json").write_text(json.dumps(build_device_report(), indent=2), encoding="utf-8")

    # Load policies (optional)
    df_model = DFPolicy(in_dim=8).to(device).eval()
    cf_model = CFPolicy(in_dim=8).to(device).eval()
    if args.df_ckpt:
        df_model.load_state_dict(torch.load(args.df_ckpt, map_location=device)["model_state"])
    if args.cf_ckpt:
        cf_model.load_state_dict(torch.load(args.cf_ckpt, map_location=device)["model_state"])

    batch = 1024
    ch = cfg["channel"]
    geo = GeometryConfig(**ch["geometry"])
    fad = FadingConfig(**ch["fading"])

    df_cfg = cfg["df"]
    cf_cfg = cfg["cf"]
    dfp = DFParams(tau=df_cfg["tau"], noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])
    cfp = CFParams(tau=cf_cfg["tau"], noise_power=cf_cfg["noise_power"], Pp=cf_cfg["power"]["Pp"], Rp0=cf_cfg["primary_qos"]["Rp0"])

    curves = {"DF(policy)": [], "CF(policy)": [], "Hybrid(rule)": [], "DF(bruteforce)": [], "CF(bruteforce)": []}

    for snr in args.snr_db:
        g = generate_gains(batch, ch["model"], device, geo=geo, fad=fad, mixed_weights=ch.get("mixed",{}).get("weights"))
        scale = 10.0 ** (snr / 10.0)
        g = {k: v * scale for k, v in g.items()}

        X = torch.stack([g[k] for k in LINKS], dim=-1)

        budgets_df = {"Ps_max": torch.full((batch,), df_cfg["power"]["Ps_max"], device=device),
                      "Pr_max": torch.full((batch,), df_cfg["power"]["Pr_max"], device=device)}
        budgets_cf = {"Ps_max": torch.full((batch,), cf_cfg["power"]["Ps_max"], device=device),
                      "Pr_max": torch.full((batch,), cf_cfg["power"]["Pr_max"], device=device)}

        act_df = df_model.predict(X, budgets_df)
        act_cf = cf_model.predict(X, budgets_cf)

        rs_df = float(torch.mean(torch.nan_to_num(torch.tensor(0.0, device=device) + 0.0)).item())  # placeholder init

        # Evaluate using baselines too (ground truth)
        df_sol = solve_df_bruteforce(g, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp, DFGrid(ps_steps=21, pr_steps=21, alpha_steps=11))
        cf_sol = solve_cf_bruteforce(g, cf_cfg["power"]["Ps_max"], cf_cfg["power"]["Pr_max"], cfp, CFGrid(ps_steps=41, pr_steps=41))
        rs_cf_b = secondary_rate_cf(g, cf_sol["ps"], cf_sol["pr"], cfp)

        # Policy RS
        from cran.physics.rates_df import secondary_rate_df
        rs_df_p = secondary_rate_df(g, act_df.ps, act_df.pr, act_df.alpha, dfp)
        rs_cf_p = secondary_rate_cf(g, act_cf.ps, act_cf.pr, cfp)

        # Hybrid (rule-based)
        hy = select_best_scheme(g,
                                df_action={"ps": act_df.ps, "pr": act_df.pr, "alpha": act_df.alpha},
                                cf_action={"ps": act_cf.ps, "pr": act_cf.pr},
                                df_params=dfp, cf_params=cfp,
                                safety=True)

        curves["DF(policy)"].append(float(rs_df_p.mean().item()))
        curves["CF(policy)"].append(float(rs_cf_p.mean().item()))
        curves["Hybrid(rule)"].append(float(torch.where(hy["scheme"]>0.5, hy["rs_df"], hy["rs_cf"]).mean().item()))
        curves["DF(bruteforce)"].append(float(df_sol["rs"].mean().item()))
        curves["CF(bruteforce)"].append(float(rs_cf_b.mean().item()))

        logger.info(f"SNR={snr:>4.1f} RS DFp={curves['DF(policy)'][-1]:.4f} CFp={curves['CF(policy)'][-1]:.4f} HY={curves['Hybrid(rule)'][-1]:.4f}")

    (apaths.results / "full_system_snr.json").write_text(json.dumps({"snr_db": args.snr_db, "rs": curves}, indent=2), encoding="utf-8")
    plot_rs_vs_snr(args.snr_db, curves, apaths.figures, title="Full-system RS vs SNR (policy + hybrid + baselines)")

if __name__ == "__main__":
    main()
