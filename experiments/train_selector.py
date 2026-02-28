# experiments/train_selector.py

"""Train the DF/CF selector (Phase 8).

Approach:
1) Generate channel batch(es)
2) Compute DF baseline RS and CF baseline RS (oracle label: argmax)
3) Train SelectorNet to predict p(DF)

Contribution:
- Hybrid DF/CF selection improves overall performance across heterogeneous channels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.physics.channels import generate_gains, GeometryConfig, FadingConfig, LINKS
from cran.physics.rates_df import DFParams
from cran.physics.rates_cf import CFParams, secondary_rate_cf
from cran.baselines.bruteforce_df import solve_df_bruteforce, DFGrid
from cran.baselines.bruteforce_cf import solve_cf_bruteforce, CFGrid

from cran.selection.selector_net import SelectorNet
from cran.selection.threshold_tuning import tune_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/cf.yaml","configs/selector.yaml"])
    ap.add_argument("--out", default="runs/train_selector")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=2048)
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

    ch = cfg["channel"]
    geo = GeometryConfig(**ch["geometry"])
    fad = FadingConfig(**ch["fading"])
    g = generate_gains(args.batch, ch["model"], device, geo=geo, fad=fad, mixed_weights=ch.get("mixed",{}).get("weights"))

    # Feature tensor: stack gains in fixed order (simple baseline feature set)
    X = torch.stack([g[k] for k in LINKS], dim=-1)

    df_cfg = cfg["df"]
    cf_cfg = cfg["cf"]
    dfp = DFParams(tau=df_cfg["tau"], noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])
    cfp = CFParams(tau=cf_cfg["tau"], noise_power=cf_cfg["noise_power"], Pp=cf_cfg["power"]["Pp"], Rp0=cf_cfg["primary_qos"]["Rp0"])

    df_sol = solve_df_bruteforce(g, df_cfg["power"]["Ps_max"], df_cfg["power"]["Pr_max"], dfp, DFGrid(ps_steps=13, pr_steps=13, alpha_steps=7))
    cf_sol = solve_cf_bruteforce(g, cf_cfg["power"]["Ps_max"], cf_cfg["power"]["Pr_max"], cfp, CFGrid(ps_steps=21, pr_steps=21))
    rs_cf = secondary_rate_cf(g, cf_sol["ps"], cf_sol["pr"], cfp)

    oracle_df = (df_sol["rs"] >= rs_cf).float()  # 1 if DF better else 0

    ds = TensorDataset(X, oracle_df)
    dl = DataLoader(ds, batch_size=512, shuffle=True)

    net = SelectorNet(in_dim=X.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    bce = torch.nn.BCELoss()

    net.train()
    for ep in range(1, args.epochs + 1):
        losses = []
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            p_df = net(xb)
            loss = bce(p_df, yb)
            loss.backward()
            opt.step()
            losses.append(loss.detach())
        logger.info(f"epoch={ep} loss={torch.stack(losses).mean().item():.6f}")

    # Threshold tuning
    net.eval()
    with torch.no_grad():
        p_df = net(X)
    t_grid = cfg["selector"]["threshold"]["grid"]
    best_t, info = tune_threshold(p_df, df_sol["rs"], rs_cf, t_grid)
    info["best_t"] = best_t

    torch.save({"model_state": net.state_dict(), "best_t": best_t}, out / "selector.pt")
    with open(out / "threshold_tuning.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    logger.info(f"best threshold={best_t} best_mean_rs={info['best_rs']:.6f}")

if __name__ == "__main__":
    main()
