"""NVIDIA GPU capability detection, precision selection, and VRAM-aware micro-batching."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class HardwareProfile:
    """Resolved hardware & execution profile for training/validation."""

    device: torch.device
    name: str
    total_vram_gb: float
    compute_capability: Optional[Tuple[int, int]]
    precision_label: str
    amp_dtype: Optional[torch.dtype]
    use_amp: bool
    use_grad_scaler: bool
    tf32_enabled: bool
    micro_batch_size: int
    grad_accum_steps: int
    num_workers: int

    @property
    def summary_str(self) -> str:
        if self.device.type == "cuda":
            idx = self.device.index if self.device.index is not None else 0
            cc_str = (
                f"SM {self.compute_capability[0]}.{self.compute_capability[1]}"
                if self.compute_capability is not None
                else "CUDA"
            )
            return f"CUDA:{idx} · {self.name} ({self.total_vram_gb:.1f} GB, {cc_str})"
        return self.device.type.upper()


def resolve_precision(
    precision: str,
    device_type: str,
    compute_capability: Optional[Tuple[int, int]] = None,
    bf16_supported: bool = False,
) -> Tuple[Optional[torch.dtype], bool, bool, bool, str]:
    """Resolve AMP dtype, scaler usage, TF32 flag, and display label.

    Returns:
        (amp_dtype, use_amp, use_grad_scaler, enable_tf32, label)
    """
    mode = precision.lower().strip()
    if mode not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError(f"Unsupported precision mode: {precision!r}. Choose auto|bf16|fp16|fp32.")

    if device_type != "cuda":
        return None, False, False, False, "FP32"

    cc_major = compute_capability[0] if compute_capability is not None else 0
    tf32_ok = cc_major >= 8

    if mode == "fp32":
        return None, False, False, tf32_ok, "FP32 + TF32" if tf32_ok else "FP32"

    if mode == "auto":
        if cc_major >= 8 and bf16_supported:
            return torch.bfloat16, True, False, True, "BF16 + TF32"
        if cc_major >= 7:
            return torch.float16, True, True, tf32_ok, "FP16 (AMP)"
        return None, False, False, False, "FP32"

    if mode == "bf16":
        if bf16_supported:
            return torch.bfloat16, True, False, tf32_ok, "BF16 + TF32" if tf32_ok else "BF16"
        # Graceful fallback on older GPUs (e.g. Volta/Turing) if bf16 explicitly requested
        if cc_major >= 7:
            return torch.float16, True, True, False, "FP16 (AMP, bf16 unsupported)"
        return None, False, False, False, "FP32 (bf16 unsupported)"

    # mode == "fp16"
    return torch.float16, True, True, tf32_ok, "FP16 + TF32" if tf32_ok else "FP16 (AMP)"


def resolve_micro_batch_size(
    batch_size: int,
    total_vram_gb: float,
    requested_micro_batch_size: Optional[int] = None,
    device_type: str = "cuda",
) -> Tuple[int, int]:
    """Choose a VRAM-safe micro-batch size while preserving logical `batch_size`.

    Returns:
        (micro_batch_size, grad_accum_steps)
    """
    bsz = max(int(batch_size), 1)
    if requested_micro_batch_size is not None and requested_micro_batch_size > 0:
        micro = min(int(requested_micro_batch_size), bsz)
    elif device_type != "cuda":
        micro = bsz
    else:
        # 10GB+ (RTX 3080/3090/4080/4090/A100/H100) runs full batch_size=64 directly.
        # 6-10GB (RTX 2060/3060Ti/3070) splits into 32; <6GB (GTX 1650/1660) into 16.
        if total_vram_gb >= 9.5:
            micro = bsz
        elif total_vram_gb >= 6.0:
            micro = min(bsz, 32)
        else:
            micro = min(bsz, 16)

    accum = max(1, int(math.ceil(bsz / float(micro))))
    return micro, accum


def resolve_num_workers(requested: int, device_type: str) -> int:
    """Resolve DataLoader worker count for zero GPU idle time without Windows spawn thrashing."""
    if device_type != "cuda":
        return 0
    if requested >= 0:
        return int(requested)
    if os.name == "nt":
        return 0
    cpu_cnt = os.cpu_count() or 2
    return min(4, max(2, cpu_cnt // 2))


def configure_hardware(
    device: torch.device,
    batch_size: int,
    precision: str = "auto",
    micro_batch_size: Optional[int] = None,
    num_workers: int = -1,
    amp_enabled: bool = True,
) -> HardwareProfile:
    """Detect GPU capabilities, apply backend flags (TF32), and build a HardwareProfile."""
    effective_precision = precision if amp_enabled else "fp32"

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        name = str(props.name)
        total_vram_gb = float(props.total_memory) / (1024.0 ** 3)
        cc = (int(props.major), int(props.minor))
        bf16_ok = bool(torch.cuda.is_bf16_supported())
    else:
        name = device.type.upper()
        total_vram_gb = 0.0
        cc = None
        bf16_ok = False

    amp_dtype, use_amp, use_scaler, tf32_on, prec_label = resolve_precision(
        precision=effective_precision,
        device_type=device.type,
        compute_capability=cc,
        bf16_supported=bf16_ok,
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = tf32_on
        torch.backends.cudnn.allow_tf32 = tf32_on

    micro, accum = resolve_micro_batch_size(
        batch_size=batch_size,
        total_vram_gb=total_vram_gb,
        requested_micro_batch_size=micro_batch_size,
        device_type=device.type,
    )
    workers = resolve_num_workers(num_workers, device.type)

    return HardwareProfile(
        device=device,
        name=name,
        total_vram_gb=total_vram_gb,
        compute_capability=cc,
        precision_label=prec_label,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        use_grad_scaler=use_scaler,
        tf32_enabled=tf32_on,
        micro_batch_size=micro,
        grad_accum_steps=accum,
        num_workers=workers,
    )


def is_oom_error(exc: BaseException) -> bool:
    """Return True if `exc` is a CUDA Out-Of-Memory error."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "cudnn" in msg)
    return False
