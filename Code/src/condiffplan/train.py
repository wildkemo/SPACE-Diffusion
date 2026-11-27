"""Training pipeline for ConDiffPlan on ResPlan (CUDA + AMP + EMA + CPU smoke test)."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from condiffplan.config import DataConfig, DiffusionConfig, ModelConfig, TrainConfig
from condiffplan.data.dataset import BucketBatchSampler, ResPlanDiffusionDataset, collate_floorplans
from condiffplan.data.preprocess import preprocess_resplan_splits
from condiffplan.hardware import HardwareProfile, configure_hardware, is_oom_error
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
    hw: HardwareProfile,
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
    batch_str = (
        f"{train_cfg.batch_size}"
        if hw.grad_accum_steps <= 1
        else f"{train_cfg.batch_size} (micro {hw.micro_batch_size} × {hw.grad_accum_steps} accum)"
    )
    disc_str = (
        f"Discrete: {model_cfg.discrete_n_layers}L/{model_cfg.discrete_num_bits}b "
        f"(train t<{diff_cfg.discrete_train_t}, sample t<{diff_cfg.discrete_sample_t})"
        if model_cfg.enable_discrete
        else "Discrete: off"
    )
    bucket_str = " · bucketed" if train_cfg.bucket_batching else ""
    rows = [
        ("Hardware", f"{hw.summary_str}  |  Precision: {hw.precision_label}"),
        (
            "Model",
            f"{model_cfg.n_layers}L × {model_cfg.d_model}d × {model_cfg.n_heads}H "
            f"({_format_params(num_params)} params)  |  T={diff_cfg.num_timesteps} ({diff_cfg.schedule})",
        ),
        ("Denoising", f"Continuous + {disc_str}"),
        (
            "Dataset",
            f"Train: {train_size:,} plans  |  Val: {val_size:,} plans  |  Batch: {batch_str}{bucket_str}  |  Workers: {hw.num_workers}",
        ),
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


class _AsyncCheckpointSaver:
    """Background atomic checkpoint writer so slow HDD I/O never freezes training."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Optional[tuple[Path, Dict[str, Any]]]]" = queue.Queue(maxsize=2)
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._worker, name="ckpt-saver", daemon=True)
        self._thread.start()

    @staticmethod
    def _to_cpu_clone(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().to(device="cpu").clone()
        if isinstance(obj, dict):
            return {k: _AsyncCheckpointSaver._to_cpu_clone(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_AsyncCheckpointSaver._to_cpu_clone(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_AsyncCheckpointSaver._to_cpu_clone(v) for v in obj)
        return obj

    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                break
            path, payload = item
            try:
                tmp_path = path.with_suffix(path.suffix + ".tmp")
                torch.save(payload, tmp_path)
                os.replace(tmp_path, path)
            except BaseException as exc:
                self._error = exc
            finally:
                self._q.task_done()

    def save(self, path: Path, payload: Dict[str, Any]) -> None:
        if self._error is not None:
            raise self._error
        cpu_payload = self._to_cpu_clone(payload)
        self._q.put((path, cpu_payload))

    def close(self) -> None:
        self._q.put(None)
        self._q.join()
        self._thread.join(timeout=60.0)
        if self._error is not None:
            raise self._error


class EMA:
    """Exponential Moving Average of model parameters using 1-kernel _foreach_lerp_."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        self._shadow_params = [p.data for p in self.shadow.parameters()]
        self._shadow_buffers = [b.data for b in self.shadow.buffers()]

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_params = [p.data for p in model.parameters()]
        if self._shadow_params:
            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(self._shadow_params, model_params, 1.0 - self.decay)
            else:
                for s_p, m_p in zip(self._shadow_params, model_params):
                    s_p.mul_(self.decay).add_(m_p, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self._shadow_buffers, model.buffers()):
            s_buf.copy_(m_buf.data)


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
    amp_dtype: Optional[torch.dtype] = None,
    use_amp: bool = False,
) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        dev_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
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

    print(f"Loading cached splits from {cache_dir}...", flush=True)
    train_dataset = ResPlanDiffusionDataset(
        train_pt, cfg=data_cfg, is_train=True, include_metadata=False
    )
    val_dataset = ResPlanDiffusionDataset(
        val_pt, cfg=data_cfg, is_train=False, include_metadata=False
    )

    hw = configure_hardware(
        device=device,
        batch_size=min(train_cfg.batch_size, len(train_dataset)),
        precision=train_cfg.precision,
        micro_batch_size=train_cfg.micro_batch_size,
        num_workers=train_cfg.num_workers,
        amp_enabled=train_cfg.amp,
    )
    if train_cfg.grad_checkpointing:
        model_cfg.grad_checkpointing = True

    effective_workers = hw.num_workers
    eff_bsz = min(train_cfg.batch_size, len(train_dataset))
    drop_last = len(train_dataset) >= train_cfg.batch_size

    loader_kwargs: Dict[str, Any] = {
        "num_workers": effective_workers,
        "collate_fn": collate_floorplans,
        "pin_memory": (device.type == "cuda"),
    }
    if train_cfg.bucket_batching and len(train_dataset) > 0:
        loader_kwargs["batch_sampler"] = BucketBatchSampler(
            lengths=train_dataset.lengths,
            batch_size=eff_bsz,
            drop_last=drop_last,
            seed=train_cfg.seed,
        )
    else:
        loader_kwargs["batch_size"] = eff_bsz
        loader_kwargs["shuffle"] = True
        loader_kwargs["drop_last"] = drop_last
    if effective_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, **loader_kwargs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=min(hw.micro_batch_size, len(val_dataset)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_floorplans,
        pin_memory=(device.type == "cuda"),
    )

    raw_model = ConDiffPlanTransformer(model_cfg).to(device)
    diffusion = ConditionalGaussianDiffusion(diff_cfg).to(device)
    ema = EMA(raw_model, decay=train_cfg.ema_decay)

    try:
        optimizer = torch.optim.Adam(
            raw_model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
            fused=(device.type == "cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.Adam(
            raw_model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )

    scaler = torch.amp.GradScaler("cuda", enabled=hw.use_grad_scaler)

    start_step = 0
    best_val_loss = float("inf")

    if resume_path is not None and Path(resume_path).exists():
        print(f"Resuming checkpoint from {resume_path}...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        missing, unexpected = raw_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing:
            print(
                f"  [resume] Freshly initialized {len(missing)} new parameter(s) "
                f"(e.g., {missing[0]}).",
                flush=True,
            )
        if unexpected:
            print(
                f"  [resume] Ignored {len(unexpected)} unexpected key(s) from checkpoint.",
                flush=True,
            )
        if "ema_state_dict" in ckpt:
            ema.shadow.load_state_dict(ckpt["ema_state_dict"], strict=False)
            if missing:
                # Sync freshly initialized parameters into EMA shadow
                ema.shadow.load_state_dict(raw_model.state_dict(), strict=True)
        if "optimizer_state_dict" in ckpt and not missing and not unexpected:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError:
                print("  [resume] Optimizer state incompatible with current model; starting fresh optimizer.", flush=True)
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))

    model: nn.Module = raw_model
    if train_cfg.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile(dynamic=True)...", flush=True)
        model = torch.compile(raw_model, dynamic=True)  # type: ignore[assignment]

    ckpt_dir = Path(train_cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest.pt"
    best_ckpt_path = ckpt_dir / "best.pt"
    ckpt_saver = _AsyncCheckpointSaver()

    def _save_ckpt(
        path: Path,
        step: int,
        val_loss: float,
        include_optimizer: bool = True,
    ) -> None:
        payload: Dict[str, Any] = {
            "step": step,
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "model_state_dict": raw_model.state_dict(),
            "ema_state_dict": ema.shadow.state_dict(),
            "model_config": model_cfg.to_dict(),
            "diffusion_config": diff_cfg.to_dict(),
            "data_config": data_cfg.to_dict(),
            "train_config": train_cfg.to_dict(),
            "corner_stats": corner_stats,
        }
        if include_optimizer:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        ckpt_saver.save(path, payload)

    model.train()
    data_iter = infinite_loader(train_loader)
    ansi = _Ansi()
    show_vram = device.type == "cuda"
    num_params = sum(p.numel() for p in raw_model.parameters())

    _print_banner(
        ansi=ansi,
        hw=hw,
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
    # Accumulate metrics on-device to avoid per-step CPU-GPU sync stalls
    running_loss_t = torch.zeros((), device=device, dtype=torch.float32)
    running_grad_t = torch.zeros((), device=device, dtype=torch.float32)
    running_count = 0
    total_steps = train_cfg.total_steps
    current_micro_bsz = hw.micro_batch_size

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
            logical_bsz = int(batch["coords"].shape[0])

            # Micro-batched forward/backward with automatic CUDA OOM recovery (zero per-step syncs)
            while True:
                optimizer.zero_grad(set_to_none=True)
                step_loss_t = torch.zeros((), device=device, dtype=torch.float32)
                try:
                    if current_micro_bsz >= logical_bsz:
                        dev_batch = {
                            k: v.to(device, non_blocking=True)
                            for k, v in batch.items()
                            if torch.is_tensor(v)
                        }
                        with torch.amp.autocast("cuda", dtype=hw.amp_dtype, enabled=hw.use_amp):
                            loss = diffusion.training_loss(model, dev_batch)
                        scaler.scale(loss).backward()
                        step_loss_t = loss.detach().float()
                    else:
                        for start in range(0, logical_bsz, current_micro_bsz):
                            end = min(start + current_micro_bsz, logical_bsz)
                            chunk_weight = float(end - start) / float(logical_bsz)
                            dev_chunk = {
                                k: v[start:end].to(device, non_blocking=True)
                                for k, v in batch.items()
                                if torch.is_tensor(v)
                            }
                            with torch.amp.autocast("cuda", dtype=hw.amp_dtype, enabled=hw.use_amp):
                                chunk_loss = diffusion.training_loss(model, dev_chunk)
                            scaler.scale(chunk_loss * chunk_weight).backward()
                            step_loss_t = step_loss_t + chunk_loss.detach().float() * chunk_weight
                    break
                except BaseException as exc:
                    if device.type == "cuda" and is_oom_error(exc) and current_micro_bsz > 1:
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        new_micro = max(1, current_micro_bsz // 2)
                        tqdm.write(
                            ansi.yellow(
                                f"  ! CUDA OOM at step {step:,} with micro_batch_size={current_micro_bsz}; "
                                f"retrying with micro_batch_size={new_micro} (effective batch={logical_bsz})"
                            )
                        )
                        current_micro_bsz = new_micro
                        continue
                    raise

            if train_cfg.grad_clip > 0:
                if hw.use_grad_scaler:
                    scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train_cfg.grad_clip)
                if torch.is_tensor(gn):
                    running_grad_t = running_grad_t + gn.detach().float()
                else:
                    running_grad_t = running_grad_t + float(gn)

            if hw.use_grad_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            ema.update(raw_model)

            running_loss_t = running_loss_t + step_loss_t
            running_count += 1
            interval_steps += 1

            if step % train_cfg.log_every == 0 or step == start_step + 1 or step == total_steps:
                now = time.time()
                avg_loss = float(running_loss_t.item()) / max(running_count, 1)
                if not math.isfinite(avg_loss):
                    raise RuntimeError(f"Non-finite training loss encountered at step {step}: {avg_loss}")
                avg_grad = float(running_grad_t.item()) / max(running_count, 1)

                postfix: Dict[str, str] = {"loss": f"{avg_loss:.4f}", "lr": f"{lr_now:.1e}"}
                if show_vram:
                    postfix["vram"] = _vram_usage_str(device)
                pbar.set_postfix(postfix, refresh=False)

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

                running_loss_t.zero_()
                running_grad_t.zero_()
                running_count = 0
                interval_t0 = now
                interval_steps = 0

            if step % train_cfg.val_every == 0 or step == total_steps:
                val_loss = evaluate_validation_loss(
                    ema.shadow,
                    diffusion,
                    val_loader,
                    device,
                    amp_dtype=hw.amp_dtype,
                    use_amp=hw.use_amp,
                )
                prev_best = best_val_loss
                is_best = val_loss <= best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    _save_ckpt(best_ckpt_path, step, val_loss, include_optimizer=False)
                _save_ckpt(latest_ckpt_path, step, val_loss, include_optimizer=True)

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
                _save_ckpt(milestone_path, step, best_val_loss, include_optimizer=False)
                tqdm.write(
                    f"{ansi.magenta('  ★ CKPT')} step {step:,} │ {ansi.dim(f'saved {milestone_path.name}')}"
                )
    finally:
        pbar.close()
        ckpt_saver.close()

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
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Optional micro-batch size for gradient accumulation (default: auto from GPU VRAM)",
    )
    parser.add_argument("--grad-checkpointing", action="store_true", help="Enable activation checkpointing for low-VRAM GPUs")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(dynamic=True) on the model")
    parser.add_argument("--no-bucket-batching", action="store_true", help="Disable length-bucketed batching")
    parser.add_argument("--num-workers", type=int, default=-1, help="DataLoader workers (-1 = auto)")
    parser.add_argument("--no-discrete", action="store_true", help="Disable HouseDiffusion discrete denoising branch")
    parser.add_argument("--discrete-bits", type=int, default=8, help="Binary bits per coordinate for discrete branch")
    parser.add_argument("--discrete-train-t", type=int, default=20, help="Train discrete branch on t < this value")
    parser.add_argument("--discrete-sample-t", type=int, default=32, help="Use discrete branch in sampling on t < this value")
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
        model_cfg.enable_discrete = not args.no_discrete
        model_cfg.discrete_num_bits = args.discrete_bits
        diff_cfg = DiffusionConfig(
            num_timesteps=20,
            discrete_train_t=min(args.discrete_train_t, 20),
            discrete_sample_t=min(args.discrete_sample_t, 20),
        )
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
            precision="fp32",
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
    model_cfg.enable_discrete = not args.no_discrete
    model_cfg.discrete_num_bits = args.discrete_bits
    model_cfg.grad_checkpointing = args.grad_checkpointing
    diff_cfg = DiffusionConfig(
        num_timesteps=args.timesteps,
        discrete_train_t=args.discrete_train_t,
        discrete_sample_t=args.discrete_sample_t,
    )
    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        lr_min=args.lr_min,
        total_steps=args.steps,
        precision=args.precision,
        micro_batch_size=args.micro_batch_size,
        grad_checkpointing=args.grad_checkpointing,
        compile_model=args.compile,
        bucket_batching=not args.no_bucket_batching,
        num_workers=args.num_workers,
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
