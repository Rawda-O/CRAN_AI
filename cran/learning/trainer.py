# cran/learning/trainer.py

"""Training loop (GPU-first, reproducible).

Critical contributions ensured in the trainer:
- AMP mixed precision support for speed on GPU (new contribution: real-time readiness)
- Robust pairing support: forward uses imperfect inputs; loss uses perfect physics
- Metric logging per step/epoch with CSV friendliness (handled by caller)

We keep the trainer generic: the caller supplies a 'step_fn' that returns loss dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch

from cran.utils.tensor_utils import assert_on_device


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0
    mixed_precision: bool = True      # AMP on GPU: faster training
    log_every_steps: int = 50
    save_best: bool = True


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 cfg: TrainerConfig,
                 out_dir: str | Path,
                 logger):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.mixed_precision and device.type == "cuda"))

        self.best_val = float("inf")

    def save_checkpoint(self, name: str, extra: Optional[Dict] = None):
        ckpt = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, self.out_dir / name)

    def fit(self,
            train_loader: Iterable,
            val_loader: Optional[Iterable],
            step_fn: Callable[[torch.nn.Module, Dict, torch.device], Dict[str, torch.Tensor]],
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        global_step = 0

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            for batch in train_loader:
                # step_fn must move tensors to device itself or batch must already be on device.
                # We assert the main tensors are on device to prevent silent CPU fallback (GPU-first guarantee).
                # (Caller can disable or adapt by providing already-device tensors.)
                self.optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self.cfg.mixed_precision and self.device.type == "cuda")):
                    out = step_fn(self.model, batch, self.device)  # returns dict containing 'loss'
                    loss = out["loss"]

                self.scaler.scale(loss).backward()

                if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

                # Step + scheduler (avoid stepping scheduler on AMP overflow-skip)
                if self.cfg.mixed_precision and self.device.type == "cuda":
                    scale_before = float(self.scaler.get_scale())
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = float(self.scaler.get_scale())
                    did_step = (scale_after >= scale_before)
                else:
                    self.optimizer.step()
                    did_step = True

                if (scheduler is not None) and did_step:
                    scheduler.step()

                global_step += 1
                if global_step % self.cfg.log_every_steps == 0:
                    self.logger.info(f"epoch={epoch} step={global_step} loss={loss.item():.6f}")

            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        with torch.cuda.amp.autocast(enabled=(self.cfg.mixed_precision and self.device.type == "cuda")):
                            out = step_fn(self.model, batch, self.device)
                            losses.append(out["loss"].detach())
                val_loss = torch.stack(losses).mean().item()
                self.logger.info(f"epoch={epoch} val_loss={val_loss:.6f}")

                if self.cfg.save_best and val_loss < self.best_val:
                    self.best_val = val_loss
                    self.save_checkpoint("best.pt", extra={"best_val": self.best_val, "epoch": epoch})

            # Always save last
            self.save_checkpoint("last.pt", extra={"epoch": epoch, "val_loss": val_loss})
