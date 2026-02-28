from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cran.utils.config_loader import merge_configs, save_config_snapshot
from cran.utils.seed import SeedConfig, set_global_seed, get_torch_device
from cran.utils.logger import build_logger

from cran.policies.df_policy import DFPolicy
from cran.learning.trainer import Trainer, TrainerConfig
from cran.learning.schedulers import build_scheduler
from cran.learning.losses import MultiObjectiveWeights, RiskConfig
from cran.learning.steps import df_step_fn, StepConfig

from cran.data.dataset import ChannelBatchConfig, ChannelBatchGenerator
from cran.data.dataloaders import FeatureConfig, LoaderConfig, make_epoch_iterator
from cran.physics.csi import ImperfectCSIConfig
from cran.physics.rates_df import DFParams

from cran.deploy.model_summary import count_parameters, model_size_mb
from cran.deploy.device_report import build_device_report
from cran.reporting.run_report import write_markdown_report
from cran.reporting.artifact_registry import ArtifactPaths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/base.yaml","configs/channel.yaml","configs/df.yaml","configs/robust.yaml","configs/multi_objective.yaml","configs/risk.yaml","configs/safety.yaml"])
    ap.add_argument("--out", default="runs/train_df")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps_per_epoch", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2048)
    args = ap.parse_args()

    cfg = merge_configs(args.configs)
    device = get_torch_device(cfg["device"]["prefer"])
    set_global_seed(SeedConfig(value=cfg["seed"]["value"],
                               deterministic=cfg["project"]["deterministic"],
                               cudnn_benchmark=cfg["device"]["cudnn_benchmark"]))
    logger = build_logger("cran-ai", cfg["logging"]["level"], f"{args.out}/run.log" if cfg["logging"]["log_to_file"] else None)

    apaths = ArtifactPaths.make(args.out)
    save_config_snapshot(cfg, apaths.root / "config_merged.yaml")

    # Model
    in_dim = 8
    model = DFPolicy(in_dim=in_dim)
    info = count_parameters(model)
    logger.info(f"DFPolicy params: {info} size~{model_size_mb(model):.2f} MB")

    # Physics params
    df_cfg = cfg["df"]
    dfp = DFParams(tau=df_cfg["tau"], noise_power=df_cfg["noise_power"], Pp=df_cfg["power"]["Pp"], Rp0=df_cfg["primary_qos"]["Rp0"])

    # Data generator (robust pairing supported)
    ch = cfg["channel"]
    bcfg = ChannelBatchConfig(
        batch_size=args.batch,
        channel_model=ch["model"],
        mixed_weights=ch.get("mixed",{}).get("weights"),
        geometry=ch.get("geometry",{}),
        fading=ch.get("fading",{}),
    )
    robust_on = bool(cfg.get("robust",{}).get("enable", False))
    icfg = None
    if robust_on:
        ncfg = cfg["robust"]["noise"]
        icfg = ImperfectCSIConfig(sigma=ncfg["sigma"],
                                  apply_to_links=tuple(ncfg["apply_to_links"]),
                                  clamp_min=ncfg["clamp_min"])
    gen = ChannelBatchGenerator(bcfg, device=device, robust=robust_on, imperfect_cfg=icfg)

    feat_cfg = FeatureConfig(include_system=False)
    loader_cfg = LoaderConfig(steps_per_epoch=args.steps_per_epoch, generalized=False, robust=robust_on)
    budgets = {"Ps_max": df_cfg["power"]["Ps_max"], "Pr_max": df_cfg["power"]["Pr_max"], "tau": df_cfg["tau"]}

    train_iter = make_epoch_iterator(gen, feat_cfg, loader_cfg, budgets)

    # Trainer + step fn
    tcfg = TrainerConfig(epochs=args.epochs, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"],
                         grad_clip_norm=cfg["optimizer"]["grad_clip_norm"],
                         mixed_precision=cfg["device"]["amp"], log_every_steps=cfg["logging"]["log_every_steps"])

    trainer = Trainer(model=model, device=device, cfg=tcfg, out_dir=apaths.checkpoints, logger=logger)
    sched = build_scheduler(trainer.optimizer, cfg["optimizer"]["scheduler"], max_steps=args.epochs * args.steps_per_epoch)

    wcfg = cfg["multi_objective"]["weights"]
    w = MultiObjectiveWeights(wr=wcfg["wr"], wq=wcfg["wq"], we=wcfg["we"], wo=wcfg["wo"])
    rcfg = cfg.get("risk",{})
    risk = RiskConfig(enable=rcfg.get("enable", False),
                      type=rcfg.get("type","cvar"),
                      cvar_alpha=rcfg.get("cvar_alpha",0.95),
                      epsilon=rcfg.get("epsilon",0.01),
                      weight=rcfg.get("weight",5.0))

    scfg = StepConfig(scheme="df", safety=cfg.get("safety",{}).get("enable", True),
                      rs_threshold=cfg["multi_objective"]["outage"]["Rs_threshold"],
                      outage_k=cfg["multi_objective"]["outage"]["k"])

    def step(model, batch, device):
        return df_step_fn(model, batch, device, dfp=dfp, w=w, risk=risk, cfg=scfg)

    # Fit (no val iterator for now)
    trainer.fit(train_loader=train_iter, val_loader=None, step_fn=step, scheduler=sched)

    # Device report + markdown report
    (apaths.results / "device_report.json").write_text(json.dumps(build_device_report(), indent=2), encoding="utf-8")
    write_markdown_report(apaths.root, "CRAN-AI DF Training Run", {
        "device": str(device),
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "robust_pairing": robust_on,
        "safety_layer": scfg.safety,
    })

if __name__ == "__main__":
    main()
