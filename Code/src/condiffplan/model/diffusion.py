"""Gaussian diffusion process with cosine noise schedule and strict fixed-corner clamping."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from condiffplan.config import DiffusionConfig
from condiffplan.model.embeddings import bits_to_coords, coords_to_bits


def _slice_batch(batch: Dict[str, torch.Tensor], mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Slice tensor entries in a collated batch along dim 0 using a boolean mask."""
    return {k: v[mask] for k, v in batch.items() if torch.is_tensor(v)}


def _discrete_settings(model: nn.Module) -> tuple[bool, int]:
    """Read (enable_discrete, discrete_num_bits) safely even if model is torch.compile wrapped."""
    base = getattr(model, "_orig_mod", model)
    enabled = bool(getattr(base, "enable_discrete", False))
    cfg = getattr(base, "cfg", None)
    num_bits = int(getattr(cfg, "discrete_num_bits", 8)) if cfg is not None else 8
    return enabled, num_bits


def make_cosine_beta_schedule(num_timesteps: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (2021), cited in ConDiffPlan Section 3.4.3 [32]."""
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64) / num_timesteps
    alphas_cumprod = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=0.999).float()


class ConditionalGaussianDiffusion(nn.Module):
    """Forward and reverse vector coordinate diffusion with fixed boundary/room clamping."""

    def __init__(self, cfg: Optional[DiffusionConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DiffusionConfig()
        self.num_timesteps = self.cfg.num_timesteps

        if self.cfg.schedule == "cosine":
            betas = make_cosine_beta_schedule(self.num_timesteps, s=self.cfg.cosine_s)
        elif self.cfg.schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, self.num_timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown schedule: {self.cfg.schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1.0))

        # Posterior q(x_{t-1} | x_t, x_0) variance & mean coefficients
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    @staticmethod
    def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
        out = a.gather(0, t)
        return out.view(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        fixed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion Eq. (4) with fixed-corner preservation (Section 3.4.2)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_ab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        x_t = sqrt_ab * x_0 + sqrt_one_minus_ab * noise
        if fixed_mask is not None:
            x_t = torch.where(fixed_mask.unsqueeze(-1), x_0, x_t)
        return x_t

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        c1 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        c2 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return c1 * x_t - c2 * eps

    def training_loss(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dual continuous + discrete L2 loss (ConDiffPlan Eq. 5 + HouseDiffusion Section 4.2).

        - Continuous branch: L2 loss between true noise and predicted noise on all timesteps.
        - Discrete branch: L2 loss between binary-encoded ground-truth x_0 and predicted bits
          on samples with t < discrete_train_t (default 20), with equal weight.
        """
        x_0 = batch["coords"]
        fixed_mask = batch["fixed_mask"]
        padding_mask = batch["padding_mask"]
        bsz = x_0.shape[0]

        if t is None:
            t = torch.randint(0, self.num_timesteps, (bsz,), device=x_0.device, dtype=torch.long)

        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0=x_0, t=t, noise=noise, fixed_mask=fixed_mask)
        pred_noise = model(x_t, t, batch)

        # Active mask: unfixed AND valid (non-padded)
        active = (~fixed_mask) & padding_mask
        active_f = active.unsqueeze(-1).to(x_0.dtype)
        sq_err = ((noise - pred_noise) ** 2) * active_f
        denom = (active_f.sum() * 2.0).clamp(min=1.0)
        loss_cont = sq_err.sum() / denom

        # HouseDiffusion discrete denoising loss (only for t < discrete_train_t)
        enable_discrete, num_bits = _discrete_settings(model)
        if enable_discrete and self.cfg.discrete_loss_weight > 0.0:
            disc_mask = t < self.cfg.discrete_train_t
            if bool(disc_mask.any().item()):
                batch_sub = _slice_batch(batch, disc_mask)
                x_0_sub = batch_sub["coords"]
                x_t_sub = x_t[disc_mask]
                t_sub = t[disc_mask]
                pred_noise_sub = pred_noise[disc_mask]
                fixed_sub = batch_sub["fixed_mask"]
                pad_sub = batch_sub["padding_mask"]

                # Compute continuous x_0 estimate and detach before discrete branch
                x_0_cont = self.predict_x0_from_eps(x_t_sub, t_sub, pred_noise_sub)
                if self.cfg.clip_denoised:
                    x_0_cont = torch.clamp(x_0_cont, -self.cfg.clip_range, self.cfg.clip_range)
                x_0_cont = torch.where(fixed_sub.unsqueeze(-1), x_0_sub, x_0_cont).detach()

                pred_bits = model.forward_discrete(x_0_cont, t_sub, batch_sub)  # type: ignore[operator]
                gt_bits = coords_to_bits(x_0_sub, num_bits=num_bits).to(pred_bits.dtype)

                active_sub = ((~fixed_sub) & pad_sub).unsqueeze(-1).to(pred_bits.dtype)
                disc_sq_err = ((gt_bits - pred_bits) ** 2) * active_sub
                disc_denom = (active_sub.sum() * float(2 * num_bits)).clamp(min=1.0)
                loss_disc = disc_sq_err.sum() / disc_denom
                return loss_cont + float(self.cfg.discrete_loss_weight) * loss_disc

        return loss_cont

    @torch.no_grad()
    def _refine_x0_discrete(
        self,
        model: nn.Module,
        x_0_pred: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply HouseDiffusion discrete branch refinement when t < discrete_sample_t."""
        fixed_mask = batch["fixed_mask"].unsqueeze(-1)
        x_fixed = batch["coords"]
        padding_mask = batch["padding_mask"].unsqueeze(-1).to(x_0_pred.dtype)

        # Ensure fixed corners are exact before entering the discrete branch
        x_0_pred = torch.where(fixed_mask, x_fixed, x_0_pred) * padding_mask

        enable_discrete, num_bits = _discrete_settings(model)
        if enable_discrete and t_index < self.cfg.discrete_sample_t:
            pred_bits = model.forward_discrete(x_0_pred, t, batch)  # type: ignore[operator]
            x_0_disc = bits_to_coords(pred_bits, num_bits=num_bits).to(x_0_pred.dtype)
            x_0_pred = torch.where(fixed_mask, x_fixed, x_0_disc) * padding_mask

        return x_0_pred

    @torch.no_grad()
    def p_sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t_index: int,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Single reverse DDPM transition p_theta(x_{t-1} | x_t) with discrete refinement and fixed clamping."""
        bsz = x_t.shape[0]
        t = torch.full((bsz,), t_index, device=x_t.device, dtype=torch.long)
        eps = model(x_t, t, batch)
        x_0_pred = self.predict_x0_from_eps(x_t, t, eps)
        if self.cfg.clip_denoised:
            x_0_pred = torch.clamp(x_0_pred, -self.cfg.clip_range, self.cfg.clip_range)
        x_0_pred = self._refine_x0_discrete(model, x_0_pred, t, t_index, batch)

        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x_0_pred + coef2 * x_t

        if t_index > 0:
            log_var = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
            z = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device, generator=generator)
            x_prev = mean + torch.exp(0.5 * log_var) * z
        else:
            x_prev = x_0_pred

        fixed_mask = batch["fixed_mask"].unsqueeze(-1)
        x_fixed = batch["coords"]
        x_prev = torch.where(fixed_mask, x_fixed, x_prev)
        return x_prev * batch["padding_mask"].unsqueeze(-1).to(x_prev.dtype)

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Full T-step DDPM reverse sampling loop Eq. (6) with fixed-element clamping."""
        x_ref = batch["coords"]
        fixed_mask = batch["fixed_mask"].unsqueeze(-1)
        padding_mask = batch["padding_mask"].unsqueeze(-1).to(x_ref.dtype)

        x_t = torch.randn(x_ref.shape, dtype=x_ref.dtype, device=x_ref.device, generator=generator)
        x_t = torch.where(fixed_mask, x_ref, x_t) * padding_mask

        for t_idx in reversed(range(self.num_timesteps)):
            x_t = self.p_sample_step(model, x_t, t_idx, batch, generator=generator)
        return x_t

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 100,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Strided DDIM reverse sampling loop with exact fixed-corner clamping at every step."""
        x_ref = batch["coords"]
        fixed_mask = batch["fixed_mask"].unsqueeze(-1)
        padding_mask = batch["padding_mask"].unsqueeze(-1).to(x_ref.dtype)
        bsz = x_ref.shape[0]

        steps = min(max(int(num_steps), 1), self.num_timesteps)
        time_seq = torch.linspace(self.num_timesteps - 1, 0, steps, dtype=torch.long, device=x_ref.device)

        x_t = torch.randn(x_ref.shape, dtype=x_ref.dtype, device=x_ref.device, generator=generator)
        x_t = torch.where(fixed_mask, x_ref, x_t) * padding_mask

        for i in range(len(time_seq)):
            t_cur = int(time_seq[i].item())
            t_prev = int(time_seq[i + 1].item()) if i + 1 < len(time_seq) else -1

            t_tensor = torch.full((bsz,), t_cur, device=x_ref.device, dtype=torch.long)
            eps = model(x_t, t_tensor, batch)
            x_0_pred = self.predict_x0_from_eps(x_t, t_tensor, eps)
            if self.cfg.clip_denoised:
                x_0_pred = torch.clamp(x_0_pred, -self.cfg.clip_range, self.cfg.clip_range)
            x_0_pred = self._refine_x0_discrete(model, x_0_pred, t_tensor, t_cur, batch)

            if t_prev < 0:
                x_t = x_0_pred
            else:
                ab_cur = self.alphas_cumprod[t_cur]
                ab_prev = self.alphas_cumprod[t_prev]
                sigma = eta * torch.sqrt((1.0 - ab_prev) / (1.0 - ab_cur) * (1.0 - ab_cur / ab_prev))
                dir_xt = torch.sqrt(torch.clamp(1.0 - ab_prev - sigma ** 2, min=0.0)) * eps
                if eta > 0.0:
                    z = torch.randn(x_ref.shape, dtype=x_ref.dtype, device=x_ref.device, generator=generator)
                else:
                    z = torch.zeros_like(x_ref)
                x_t = torch.sqrt(ab_prev) * x_0_pred + dir_xt + sigma * z

            x_t = torch.where(fixed_mask, x_ref, x_t) * padding_mask

        return x_t
