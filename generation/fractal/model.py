"""Conditional DDPM for the monophonic detailer (v2).

Denoises a detailed-melody roll conditioned on (a) the coarse roll [concatenated], and (b) a global
conditioning vector built from density + character + harmony embeddings, with classifier-free guidance
(CFG): training randomly drops the global condition to a learned null; sampling extrapolates between
the conditioned and null predictions. The velocity channel is just another (continuous) roll channel.
Sampling supports inpaint-anchoring (re-impose the coarse structural cells, RePaint-style) so the given
line is preserved.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from generation.fractal.dataset import N_CHAR, N_HARM
from generation.fractal.roll import C, P, REST, ONSET, VEL, T


def make_schedule(steps: int = 200, beta0: float = 1e-4, beta1: float = 2e-2) -> dict:
    betas = torch.linspace(beta0, beta1, steps)
    alphas = 1.0 - betas
    return {"steps": steps, "betas": betas, "alphas": alphas, "abar": torch.cumprod(alphas, 0)}


def _timestep_emb(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class FiLMResBlock(nn.Module):
    def __init__(self, ch, cond_dim, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch); self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, ch); self.conv2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.film = nn.Linear(cond_dim, ch * 2)

    def forward(self, x, cond):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(cond)[:, :, None].chunk(2, 1)
        h = self.conv2(F.silu(self.norm2(h * (1 + scale) + shift)))
        return x + h


class Denoiser(nn.Module):
    def __init__(self, width=128, depth=10, cond_dim=128):
        super().__init__()
        self.cond_dim = cond_dim
        self.inp = nn.Conv1d(2 * C, width, 1)
        self.temb = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.char_emb = nn.Embedding(N_CHAR, cond_dim, padding_idx=0)
        self.harm_emb = nn.Embedding(N_HARM, cond_dim, padding_idx=0)
        self.dens = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.null_cond = nn.Parameter(torch.zeros(cond_dim))          # CFG null condition
        self.blocks = nn.ModuleList([FiLMResBlock(width, cond_dim, dilation=2 ** (i % 5)) for i in range(depth)])
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, C, 1))

    def _global_cond(self, char, harm, density, drop):
        cv = self.char_emb(char).sum(1) + self.harm_emb(harm).sum(1) + self.dens(density[:, None].float())
        return torch.where(drop[:, None], self.null_cond[None], cv)   # CFG: drop -> null

    def forward(self, x, coarse, t, char, harm, density, drop):
        cond = self.temb(_timestep_emb(t, self.cond_dim)) + self._global_cond(char, harm, density, drop)
        h = self.inp(torch.cat([x, coarse], 1))
        for b in self.blocks:
            h = b(h, cond)
        return self.out(h)


def p_losses(model, x0, coarse, char, harm, density, sched, drop_p=0.15) -> torch.Tensor:
    abar = sched["abar"].to(x0.device)
    t = torch.randint(0, sched["steps"], (x0.size(0),), device=x0.device)
    ab = abar[t][:, None, None]
    noise = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    drop = torch.rand(x0.size(0), device=x0.device) < drop_p
    return F.mse_loss(model(xt, coarse, t, char, harm, density, drop), noise)


@torch.no_grad()
def sample(model, coarse, char, harm, density, sched, *, guidance=2.0, anchor=None) -> torch.Tensor:
    """Reverse diffusion with CFG. ``anchor`` = (mask, x0) re-imposes the given structural cells
    (RePaint): mask (1,C,T) marks known channels/cells, x0 their clean [-1,1] values."""
    dev = coarse.device
    betas, alphas, abar = (sched[k].to(dev) for k in ("betas", "alphas", "abar"))
    B = coarse.size(0)
    nodrop = torch.zeros(B, dtype=torch.bool, device=dev)
    yesdrop = torch.ones(B, dtype=torch.bool, device=dev)
    x = torch.randn(B, C, T, device=dev)
    for i in reversed(range(sched["steps"])):
        t = torch.full((B,), i, device=dev, dtype=torch.long)
        if anchor is not None:                                       # re-impose known cells, noised to step i
            mask, x0k = anchor
            x = x * (1 - mask) + (abar[i].sqrt() * x0k + (1 - abar[i]).sqrt() * torch.randn_like(x)) * mask
        pc = model(x, coarse, t, char, harm, density, nodrop)
        pn = model(x, coarse, t, char, harm, density, yesdrop)
        pred = pn + guidance * (pc - pn)                            # classifier-free guidance
        x = (x - betas[i] / (1 - abar[i]).sqrt() * pred) / alphas[i].sqrt()
        if i > 0:
            x = x + betas[i].sqrt() * torch.randn_like(x)
    if anchor is not None:
        mask, x0k = anchor
        x = x * (1 - mask) + x0k * mask
    return x


def discretize(x: torch.Tensor) -> torch.Tensor:
    """Continuous sample (C,T) -> clean roll: pitch+rest argmax, onset thresholded, velocity kept
    continuous (un-relaxed to [0,1])."""
    roll = torch.zeros_like(x)
    win = x[: P + 1].argmax(0)                                       # pitch channels + rest
    roll[win, torch.arange(x.size(1))] = 1.0
    roll[ONSET] = (x[ONSET] > 0.0).float() * (win < P).float()
    roll[VEL] = ((x[VEL] + 1) / 2).clamp(0, 1) * (win < P).float()   # un-relax; 0 on rests
    return roll
