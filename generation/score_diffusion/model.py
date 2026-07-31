"""Self-contained continuous DDPM denoiser over the whole-score onset roll.

The model is deliberately label-free: it learns ``p(score-window)`` from the full corpus by
the standard denoising objective (predict the noise added to ``x0``). Sampling is ancestral
DDPM. Two capabilities cover the "successively refined complete score" idea:

* **Unconditional generation** -- start from Gaussian noise and denoise to a full window.
* **Coarse-anchored refinement / infilling** -- :func:`sample` accepts a RePaint-style
  ``anchor = (known_mask, x0)`` that re-imposes known cells at every reverse step, so the model
  fills only the unknown region. This is what turns a coarse sketch (or a masked score) into a
  finished one, and it is also how the evaluation harness measures reconstruction.

Everything operates on ``(B, CHANNELS, STEPS)`` tensors in ``[-1, 1]``; discretization back to a
``{0,1}`` onset roll is a threshold at 0. Musical materialization stays in Ruby Partitura.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from generation.score_diffusion.roll import CHANNELS, STEPS


def make_schedule(steps=200, cosine_s=0.008) -> dict:
    """Cosine DDPM schedule whose terminal state is near-pure noise at any step count.

    A fixed linear ``1e-4..2e-2`` schedule only reaches ``alpha_bar≈0.13`` at 200 steps
    (and ``≈0.82`` at 20), making inference from pure noise inconsistent with training.
    The cosine construction reaches the noise endpoint even for short smoke schedules.
    """
    x = torch.linspace(0, steps, steps + 1)
    abar_curve = torch.cos(((x / steps + cosine_s) / (1 + cosine_s)) * math.pi * 0.5) ** 2
    abar_curve = abar_curve / abar_curve[0]
    betas = (1 - abar_curve[1:] / abar_curve[:-1]).clamp(1e-5, 0.999)
    alphas = 1.0 - betas
    return {"steps": steps, "betas": betas, "alphas": alphas, "abar": torch.cumprod(alphas, 0)}


def _timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class FiLMResBlock(nn.Module):
    """Dilated Conv1d residual block, FiLM-modulated by the timestep embedding."""

    def __init__(self, ch, cond_dim, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.film = nn.Linear(cond_dim, ch * 2)

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond)[:, :, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h) * (1 + scale) + shift))
        return x + h


class ScoreDenoiser(nn.Module):
    """1-D UNet-free residual denoiser: project 1056 channels down, stack dilated blocks, project back."""

    def __init__(self, channels=CHANNELS, width=192, depth=8, cond_dim=128):
        super().__init__()
        self.channels = channels
        self.cond_dim = cond_dim
        self.inp = nn.Conv1d(channels, width, 1)
        self.temb = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.blocks = nn.ModuleList(
            [FiLMResBlock(width, cond_dim, dilation=2 ** (i % 5)) for i in range(depth)]
        )
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, channels, 1))

    def forward(self, x, t):
        cond = self.temb(_timestep_emb(t, self.cond_dim))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, cond)
        return self.out(h)


def p_losses(model, x0, sched, *, active_weight=32.0, x0_weight=0.1) -> torch.Tensor:
    """DDPM noise MSE plus a sparse-active reconstruction term.

    Only ~0.2% of cells are active onsets. Pure epsilon MSE can minimize its loss while
    effectively treating every cell as background. The auxiliary term reconstructs ``x0`` from
    the predicted epsilon and upweights active cells, normalized by total weight so its scale is
    stable. It does not use labels beyond the self-supervised clean window.
    """
    abar = sched["abar"].to(x0.device)
    t = torch.randint(0, sched["steps"], (x0.size(0),), device=x0.device)
    ab = abar[t][:, None, None]
    noise = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    pred_noise = model(xt, t)
    noise_loss = F.mse_loss(pred_noise, noise)
    x0_pred = (xt - (1 - ab).sqrt() * pred_noise) / ab.sqrt().clamp_min(1e-4)
    weights = torch.where(x0 > 0, torch.as_tensor(active_weight, device=x0.device), 1.0)
    active_loss = ((x0_pred.clamp(-3, 3) - x0).square() * weights).sum() / weights.sum()
    return noise_loss + x0_weight * active_loss


@torch.no_grad()
def sample(model, sched, *, n=1, shape=None, anchor=None, device="cpu", generator=None):
    """Ancestral DDPM sampling.

    ``anchor = (known_mask, x0)`` re-imposes known cells (RePaint): ``known_mask`` is ``1`` on the
    cells whose clean values ``x0`` are given and should be preserved; the complement is generated.
    Returns a ``(n, CHANNELS, STEPS)`` field in ``[-1, 1]``.
    """
    shape = shape if shape is not None else (n, model.channels, STEPS)
    betas, alphas, abar = (sched[k].to(device) for k in ("betas", "alphas", "abar"))
    x = torch.randn(shape, device=device, generator=generator)
    for i in reversed(range(sched["steps"])):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        if anchor is not None:
            mask, x0k = anchor
            noised_known = abar[i].sqrt() * x0k + (1 - abar[i]).sqrt() * torch.randn(shape, device=device, generator=generator)
            x = x * (1 - mask) + noised_known * mask
        eps = model(x, t)
        x = (x - betas[i] / (1 - abar[i]).sqrt() * eps) / alphas[i].sqrt()
        if i > 0:
            x = x + betas[i].sqrt() * torch.randn(shape, device=device, generator=generator)
    if anchor is not None:
        mask, x0k = anchor
        x = x * (1 - mask) + x0k * mask
    return x


def discretize(x, threshold=0.0) -> torch.Tensor:
    """Continuous field -> {0,1} onset roll."""
    return (x > threshold).float()


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
