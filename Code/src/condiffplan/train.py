"""Training pipeline for ConDiffPlan on ResPlan (CUDA + AMP + EMA + CPU smoke test)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from condiffplan.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from condiffplan.data.dataset import ResPlanDiffusionDataset, collate_floorplans
from condiffplan.data.preprocess import preprocess_resplan_splits
from condiffplan.model.diffusion import ConditionalGaussianDiffusion
from condiffplan.model.transformer import ConDiffPlanTransformer
from condiffplan.pipeline import resolve_device


class EMA:
    """Exponential Moving Average of model parameters for higher sampling stability."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.data.copy_(m_buf.data)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infinite_loader(loader: DataLoader) -> Iterator[Dict[str, Any]]:
    while True:
        for batch in loader:
            yield batch


def exponential_lr(step: int, total_steps: int, lr_start: float, lr_end: float) -> float:
    """Exponential learning rate falloff from lr_start to lr_end over total_steps (Section 4.2)."""
    if total_steps <= 0 or lr_start <= 0 or lr_end <= 0:
        return lr_start
    progress = min(max(step / float(total_steps), 0.0), 1.0)
    return lr_start * ((lr_end / lr_start) ** progress)


@torch.no_grad()
def evaluate_validation_loss(
    model: nn.Module,
    diffusion: ConditionalGaussianDiffusion,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 25,
) -> float:
    model.eval()
    losses = []
    tensor_keys = [
        "coords",
        "elem_type",
        "elem_idx",
        "corner_idx",
        "next_corner_idx",
        "fixed_mask",
        "norm_area",
        "has_area",
        "padding_mask",
        "room_mask",
        "rel_mask",
        "rel_edge_type",
    ]
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        dev_batch = {k: batch[k].to(device) for k in tensor_keys}
        loss = diffusion.training_loss(model, dev_batch)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else 0.0


def run_training(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    diff_cfg: DiffusionConfig,
    train_cfg: TrainConfig,
    device_str: str = "auto",
    resume_path: Optional[str] = None,
    limit_per_split: Optional[int] = None,
) -> Path:
    """Execute training loop and return the path to the latest saved checkpoint."""
    set_seed(train_cfg.seed)
    device = resolve_device(device_str)

    cache_dir = Path(data_cfg.cache_dir)
    train_pt = cache_dir / "train.pt"
    val_pt = cache_dir / "val.pt"
    stats_path = cache_dir / "corner_stats.json"

    if not train_pt.exists() or not val_pt.exists():
        preprocess_resplan_splits(
            cfg=data_cfg,
            limit_per_split=limit_per_split,
            show_progress=True,
        )

    corner_stats = None
    if stats_path.exists():
        with open(stats_path, "r") as f:
            raw_stats = json.load(f)
        corner_stats = {t: {int(k): int(v) for k, v in d.items()} for t, d in raw_stats.items()}

    train_dataset = ResPlanDiffusionDataset(train_pt, cfg=data_cfg, is_train=True)
    val_dataset = ResPlanDiffusionDataset(val_pt, cfg=data_cfg, is_train=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(train_cfg.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=train_cfg.num_workers if device.type == "cuda" else 0,
        collate_fn=collate_floorplans,
        pin_memory=(device.type == "cuda"),
        drop_last=(len(train_dataset) >= train_cfg.batch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(train_cfg.batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_floorplans,
    )

    model = ConDiffPlanTransformer(model_cfg).to(device)
    diffusion = ConditionalGaussianDiffusion(diff_cfg).to(device)
    ema = EMA(model, decay=train_cfg.ema_decay)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    use_amp = bool(train_cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_step = 0
    best_val_loss = float("inf")

    if resume_path is not None and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest.pt"
    best_ckpt_path = ckpt_dir / "best.pt"

    def _save_ckpt(path: Path, step: int, val_loss: float) -> None:
        payload = {
            "step": step,
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.shadow.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_cfg.to_dict(),
            "diffusion_config": diff_cfg.to_dict(),
            "data_config": data_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
            "corner_stats": corner_stats,
        }
        torch.save(payload, path)

    tensor_keys = [
        "coords",
        "elem_type",
        "elem_idx",
        "corner_idx",
        "next_corner_idx",
        "fixed_mask",
        "norm_area",
        "has_area",
        "padding_mask",
        "room_mask",
        "rel_mask",
        "rel_edge_type",
    ]

    model.train()
    data_iter = infinite_loader(train_loader)
    t0 = time.time()
    running_loss = 0.0
    running_count = 0

    for step in range(start_step + 1, train_cfg.total_steps + 1):
        lr_now = exponential_lr(step, train_cfg.total_steps, train_cfg.lr, train_cfg.lr_min)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        batch = next(data_iter)
        dev_batch = {k: batch[k].to(device, non_blocking=True) for k in tensor_keys}

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = diffusion.training_loss(model, dev_batch)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss encountered at step {step}: {loss.item()}")

        scaler.scale(loss).backward()
        if train_cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        running_loss += float(loss.item())
        running_count += 1

        if step % train_cfg.log_every == 0 or step == 1 or step == train_cfg.total_steps:
            avg_loss = running_loss / max(running_count, 1)
            elapsed = time.time() - t0
            print(
                f"[Step {step}/{train_cfg.total_steps}] "
                f"loss={avg_loss:.5f} lr={lr_now:.2e} device={device.type} ({elapsed:.1f}s)"
            )
            running_loss = 0.0
            running_count = 0

        if step % train_cfg.val_every == 0 or step == train_cfg.total_steps:
            val_loss = evaluate_validation_loss(ema.shadow, diffusion, val_loader, device)
            print(f"  -> Validation loss at step {step}: {val_loss:.5f}")
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                _save_ckpt(best_ckpt_path, step, val_loss)
            _save_ckpt(latest_ckpt_path, step, val_loss)

        if step % train_cfg.save_every == 0:
            _save_ckpt(ckpt_dir / f"step_{step}.pt", step, best_val_loss)

    return latest_ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ConDiffPlan conditional vector diffusion model.")
    parser.add_argument("--data", type=str, default="../Dataset/ResPlan.pkl")
    parser.add_argument("--split", type=str, default="../Dataset/split.json")
    parser.add_argument("--cache-dir", type=str, default="data_cache")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--preset", type=str, default="default", choices=["default", "small", "tiny"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-test", action="store_true", help="Fast CPU smoke test on 32 plans for 5 steps")
    args = parser.parse_args()

    if args.smoke_test:
        data_cfg = DataConfig(
            data_path=args.data,
            split_path=args.split,
            cache_dir=args.cache_dir,
        )
        model_cfg = ModelConfig.preset("tiny")
        diff_cfg = DiffusionConfig(num_timesteps=20)
        train_cfg = TrainConfig(
            batch_size=8,
            lr=1e-3,
            lr_min=1e-4,
            total_steps=5,
            log_every=1,
            val_every=5,
            save_every=5,
            checkpoint_dir=args.checkpoint_dir,
            seed=args.seed,
            amp=False,
        )
        ckpt_path = run_training(
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            diff_cfg=diff_cfg,
            train_cfg=train_cfg,
            device_str=args.device,
            resume_path=args.resume,
            limit_per_split=32,
        )
        print(f"Smoke test completed. Checkpoint saved at: {ckpt_path}")
        return

    data_cfg = DataConfig(
        data_path=args.data,
        split_path=args.split,
        cache_dir=args.cache_dir,
    )
    model_cfg = ModelConfig.preset(args.preset)
    diff_cfg = DiffusionConfig(num_timesteps=args.timesteps)
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        total_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )
    ckpt_path = run_training(
        data_cfg=data_cfg,
        model_cfg=model_cfg,
        diff_cfg=diff_cfg,
        train_cfg=train_cfg,
        device_str=args.device,
        resume_path=args.resume,
    )
    print(f"Training completed. Latest checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
