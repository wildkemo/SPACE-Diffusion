"""Training pipeline for ConDiffPlan on ResPlan (CUDA + AMP + EMA + CPU smoke test)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from condiffplan.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from condiffplan.data.dataset import ResPlanDiffusionDataset, collate_floorplans
from condiffplan.data.preprocess import preprocess_resplan_splits
from condiffplan.model.diffusion import ConditionalGaussianDiffusion
from condiffplan.model.transformer import ConDiffPlanTransformer
from condiffplan.pipeline import resolve_device


class _Ansi:
    """Lightweight ANSI styling helper with automatic TTY/Colab/NO_COLOR detection."""

    def __init__(self, enabled: Optional[bool] = None) -> None:
        if enabled is None:
            no_color = bool(os.environ.get("NO_COLOR"))
            in_notebook = bool(
                os.environ.get("COLAB_GPU")
                or os.environ.get("COLAB_RELEASE_TAG")
                or os.environ.get("JPY_PARENT_PID")
            )
            enabled = (not no_color) and (sys.stdout.isatty() or in_notebook)
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def magenta(self, text: str) -> str:
        return self._wrap("35", text)

    def bold_cyan(self, text: str) -> str:
        return self._wrap("1;36", text)

    def bold_green(self, text: str) -> str:
        return self._wrap("1;32", text)


def _format_duration(seconds: float) -> str:
    """Format seconds into a compact human-readable duration (e.g., 1h 23m 45s)."""
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    total = int(round(seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours:02d}h {mins:02d}m"
    if hours > 0:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    if mins > 0:
        return f"{mins}m {secs:02d}s"
    return f"{secs}s"


def _format_params(num_params: int) -> str:
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    return str(num_params)


def _device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        return f"CUDA:{idx} · {props.name} ({vram_gb:.1f} GB)"
    return device.type.upper()


def _vram_usage_str(device: torch.device) -> str:
    if device.type != "cuda":
        return "—"
    alloc_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
    peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return f"{alloc_gb:.2f}/{peak_gb:.2f}G"


def _print_banner(
    ansi: _Ansi,
    device: torch.device,
    use_amp: bool,
    model_cfg: ModelConfig,
    diff_cfg: DiffusionConfig,
    train_cfg: TrainConfig,
    num_params: int,
    train_size: int,
    val_size: int,
    start_step: int,
    resume_path: Optional[str],
) -> None:
    title = " ConDiffPlan · Vector Floorplan Diffusion Training "
    rows = [
        ("Hardware", f"{_device_summary(device)}  |  Precision: {'AMP (FP16)' if use_amp else 'FP32'}"),
        (
            "Model",
            f"{model_cfg.n_layers}L × {model_cfg.d_model}d × {model_cfg.n_heads}H "
            f"({_format_params(num_params)} params)  |  T={diff_cfg.num_timesteps} ({diff_cfg.schedule})",
        ),
        ("Dataset", f"Train: {train_size:,} plans  |  Val: {val_size:,} plans  |  Batch: {train_cfg.batch_size}"),
        (
            "Schedule",
            f"Steps: {start_step:,} → {train_cfg.total_steps:,}  |  "
            f"LR: {train_cfg.lr:.1e} → {train_cfg.lr_min:.1e}  |  EMA: {train_cfg.ema_decay}",
        ),
        (
            "Checkpoints",
            f"{train_cfg.checkpoint_dir}  (val every {train_cfg.val_every:,}, save every {train_cfg.save_every:,})",
        ),
    ]
    if resume_path is not None and start_step > 0:
        rows.append(("Resumed", f"{resume_path} (step {start_step:,})"))

    max_row_len = max(len(f" {label:<13} {val} ") for label, val in rows)
    width = max(78, max_row_len + 2, len(title) + 4)
    border_top = f"╭{'─' * (width - 2)}╮"
    border_mid = f"├{'─' * (width - 2)}┤"
    border_bot = f"╰{'─' * (width - 2)}╯"

    def _row(label: str, value: str) -> str:
        plain_label = f"{label:<13}"
        content = f" {plain_label} {value}"
        pad = max(width - 2 - len(content), 0)
        return f"{ansi.dim('│')} {ansi.cyan(plain_label)} {value}{' ' * pad}{ansi.dim('│')}"

    title_pad = max(width - 2 - len(title), 0)
    left_pad = title_pad // 2
    right_pad = title_pad - left_pad

    lines = [
        "",
        ansi.dim(border_top),
        f"{ansi.dim('│')}{' ' * left_pad}{ansi.bold_cyan(title)}{' ' * right_pad}{ansi.dim('│')}",
        ansi.dim(border_mid),
        *(_row(label, val) for label, val in rows),
        ansi.dim(border_bot),
    ]
    print("\n".join(lines), flush=True)


def _table_header(ansi: _Ansi, show_vram: bool) -> str:
    cols = [
        f"{'Step':>17}",
        f"{'Prog':>6}",
        f"{'Loss':>9}",
        f"{'Grad':>7}",
        f"{'LR':>9}",
        f"{'Speed':>10}",
    ]
    if show_vram:
        cols.append(f"{'VRAM':>11}")
    cols.extend([f"{'Elapsed':>10}", f"{'ETA':>10}"])
    header = " │ ".join(cols)
    sep = "─" * len(header)
    return f"{ansi.bold(header)}\n{ansi.dim(sep)}"


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
    ansi = _Ansi()
    show_vram = device.type == "cuda"
    num_params = sum(p.numel() for p in model.parameters())

    _print_banner(
        ansi=ansi,
        device=device,
        use_amp=use_amp,
        model_cfg=model_cfg,
        diff_cfg=diff_cfg,
        train_cfg=train_cfg,
        num_params=num_params,
        train_size=len(train_dataset),
        val_size=len(val_dataset),
        start_step=start_step,
        resume_path=resume_path,
    )
    print(_table_header(ansi, show_vram=show_vram), flush=True)

    t0 = time.time()
    interval_t0 = t0
    interval_steps = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    running_count = 0
    total_steps = train_cfg.total_steps

    pbar = tqdm(
        range(start_step + 1, total_steps + 1),
        initial=start_step,
        total=total_steps,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )

    try:
        for step in pbar:
            lr_now = exponential_lr(step, total_steps, train_cfg.lr, train_cfg.lr_min)
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
            grad_norm_val = 0.0
            if train_cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                grad_norm_val = float(gn.item()) if torch.is_tensor(gn) else float(gn)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            loss_val = float(loss.item())
            running_loss += loss_val
            running_grad_norm += grad_norm_val
            running_count += 1
            interval_steps += 1

            postfix: Dict[str, str] = {"loss": f"{loss_val:.4f}", "lr": f"{lr_now:.1e}"}
            if show_vram:
                postfix["vram"] = _vram_usage_str(device)
            pbar.set_postfix(postfix, refresh=False)

            if step % train_cfg.log_every == 0 or step == start_step + 1 or step == total_steps:
                now = time.time()
                avg_loss = running_loss / max(running_count, 1)
                avg_grad = running_grad_norm / max(running_count, 1)
                elapsed = now - t0
                steps_done = step - start_step
                interval_dt = max(now - interval_t0, 1e-6)
                speed = interval_steps / interval_dt
                avg_speed = steps_done / max(elapsed, 1e-6)
                eta = (total_steps - step) / max(avg_speed, 1e-9)
                pct = 100.0 * step / max(total_steps, 1)

                step_str = f"{step:>7,}/{total_steps:<9,}"
                row_cols = [
                    ansi.bold(step_str),
                    f"{pct:5.1f}%",
                    ansi.yellow(f"{avg_loss:9.5f}"),
                    f"{avg_grad:7.3f}" if train_cfg.grad_clip > 0 else f"{'—':>7}",
                    ansi.dim(f"{lr_now:9.2e}"),
                    f"{speed:5.1f} it/s",
                ]
                if show_vram:
                    row_cols.append(ansi.magenta(f"{_vram_usage_str(device):>11}"))
                row_cols.extend([
                    ansi.dim(f"{_format_duration(elapsed):>10}"),
                    ansi.cyan(f"{_format_duration(eta):>10}"),
                ])
                tqdm.write(" │ ".join(row_cols))

                running_loss = 0.0
                running_grad_norm = 0.0
                running_count = 0
                interval_t0 = now
                interval_steps = 0

            if step % train_cfg.val_every == 0 or step == total_steps:
                val_loss = evaluate_validation_loss(ema.shadow, diffusion, val_loader, device)
                prev_best = best_val_loss
                is_best = val_loss <= best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    _save_ckpt(best_ckpt_path, step, val_loss)
                _save_ckpt(latest_ckpt_path, step, val_loss)

                if is_best:
                    delta_str = (
                        f" (↓ {prev_best - val_loss:.5f} new best!)"
                        if math.isfinite(prev_best)
                        else " (initial best)"
                    )
                    tag = ansi.bold_green("  ✓ VAL ")
                    val_txt = ansi.bold_green(f"{val_loss:.5f}{delta_str}")
                    saved_txt = "saved best.pt, latest.pt"
                else:
                    tag = ansi.cyan("  • VAL ")
                    val_txt = f"{val_loss:.5f} (best: {best_val_loss:.5f})"
                    saved_txt = "saved latest.pt"

                tqdm.write(f"{tag} step {step:,} │ val_loss {val_txt} │ {ansi.dim(saved_txt)}")

            if step % train_cfg.save_every == 0:
                milestone_path = ckpt_dir / f"step_{step}.pt"
                _save_ckpt(milestone_path, step, best_val_loss)
                tqdm.write(
                    f"{ansi.magenta('  ★ CKPT')} step {step:,} │ {ansi.dim(f'saved {milestone_path.name}')}"
                )
    finally:
        pbar.close()

    total_elapsed = time.time() - t0
    completed_steps = max(total_steps - start_step, 0)
    avg_it_s = completed_steps / max(total_elapsed, 1e-6)
    summary_line = "─" * 78
    best_str = f"{best_val_loss:.5f}" if math.isfinite(best_val_loss) else "n/a"
    print(
        "\n".join(
            [
                ansi.dim(summary_line),
                f"{ansi.bold_green('✓ Training Complete')}  │  "
                f"Steps: {total_steps:,}  │  "
                f"Best Val Loss: {ansi.bold_green(best_str)}  │  "
                f"Time: {_format_duration(total_elapsed)} ({avg_it_s:.1f} it/s)",
                f"  {ansi.dim('Latest:')} {latest_ckpt_path}    {ansi.dim('Best:')} {best_ckpt_path}",
                ansi.dim(summary_line),
            ]
        ),
        flush=True,
    )

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
