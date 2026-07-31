"""Discrete (Bernoulli) diffusion over the onset roll -- the phase-11 reformulation.

The phase-10 finding was that a continuous Gaussian DDPM is the wrong prior for a 99.8%-empty
*binary* onset grid: it modelled the background and decoded ~200x too dense. The onset roll is
Bernoulli, so here corruption and generation are discrete.

Forward process -- **survival to silence.** Each active onset in the clean roll ``x0`` survives to
step ``t`` independently with probability ``abar[t]`` and is otherwise absorbed to silence (0).
Inactive cells stay 0. As ``t -> T`` the survival probability ``-> 0``, so the corrupted roll ``xt``
decays to an all-silent grid: the model's prior is silence, and generation *adds* onsets.

Reverse process -- **reveal from silence.** Starting from an empty grid, at each reverse step the
model predicts per-cell occupancy ``p(x0=1 | xt, t)`` (occupancy logits) and reveals a fraction of
the still-silent cells according to the schedule increment. Cells that are already active stay
active (survival is monotone). At the final step the remaining cells are decided directly by the
predicted occupancy. This is exactly the fractal "mask-reveal ladder", now over the full note
surface, and its decoded density is controlled by the model's confidence rather than a fixed
threshold on Gaussian noise.

Materialization stays in Ruby Partitura; this module only produces ``{0,1}`` rolls.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generation.score_diffusion.dataset import from_x0  # noqa: F401  (kept for API symmetry)
from generation.score_diffusion.model import FiLMResBlock, _timestep_emb
from generation.score_diffusion.roll import CHANNELS, STEPS


def make_survival_schedule(steps=200, cosine_s=0.008) -> dict:
    """Cosine survival probabilities ``abar[t]`` decreasing from ~1 (clean) to ~0 (silence)."""
    x = torch.linspace(0, steps, steps + 1)
    curve = torch.cos(((x / steps + cosine_s) / (1 + cosine_s)) * math.pi * 0.5) ** 2
    curve = curve / curve[0]
    abar = curve[1:].clamp(1e-6, 1.0)          # abar[t], t = 0..steps-1 ; abar[-1] ~ 0
    return {"steps": steps, "abar": abar}


class OccupancyDenoiser(nn.Module):
    """Predicts clean per-cell onset occupancy logits from a partially-revealed binary roll."""

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

    def forward(self, xt, t):
        # xt is {0,1}; center to [-1,1] so an empty grid is not a zero input.
        cond = self.temb(_timestep_emb(t, self.cond_dim))
        h = self.inp(xt * 2.0 - 1.0)
        for b in self.blocks:
            h = b(h, cond)
        return self.out(h)                      # occupancy logits


def q_sample(x0, t, sched, generator=None):
    """Survival corruption: keep each active cell with prob ``abar[t]``, else absorb to silence."""
    abar = sched["abar"].to(x0.device)
    keep_p = abar[t][:, None, None]
    u = torch.rand(x0.shape, device=x0.device, generator=generator)
    survive = (u < keep_p).float()
    return x0 * survive


def p_losses(model, x0, sched, *, pos_weight=50.0, generator=None) -> torch.Tensor:
    """Class-balanced BCE between predicted occupancy and the clean roll, at a random step.

    Cells already surviving in ``xt`` are trivially positive; the loss is dominated by the cells
    the model must *recover*, so we only score cells that are silent in ``xt`` (the reveal target)
    plus keep a light term on survivors to preserve them.
    """
    t = torch.randint(0, sched["steps"], (x0.size(0),), device=x0.device)
    xt = q_sample(x0, t, sched, generator=generator)
    logits = model(xt, t)
    pw = torch.as_tensor(pos_weight, device=x0.device)
    # Recover target on the silent (revealable) cells; survivors get a small preservation weight.
    revealable = (xt < 0.5)
    per_cell = F.binary_cross_entropy_with_logits(logits, x0, pos_weight=pw, reduction="none")
    reveal_loss = (per_cell * revealable).sum() / revealable.sum().clamp_min(1.0)
    keep_loss = (per_cell * (~revealable)).sum() / (~revealable).sum().clamp_min(1.0)
    return reveal_loss + 0.1 * keep_loss


@torch.no_grad()
def sample(model, sched, *, n=1, shape=None, anchor=None, device="cpu", generator=None,
           hard=True, threshold=None):
    """Predict-x0 ancestral sampling for the survival process.

    Standard discrete-diffusion reverse step: at level ``t`` predict clean occupancy
    ``p(x0=1|xt)``, sample an estimate ``x0_hat``, then re-corrupt it to the next level ``t-1``
    (survive each estimated onset with prob ``abar[t-1]``). This keeps the working roll a sparse,
    in-distribution survival subset at every step, which avoids the density cascade that a monotone
    "accumulate reveals" sampler suffers when the model goes out-of-distribution.

    ``anchor = (context_active,)`` keeps given onset cells present every step (RePaint-style
    conditioning for infilling). ``threshold`` switches x0 estimation from Bernoulli sampling to a
    deterministic occupancy threshold, which is the calibrated decode for such a sparse grid (a
    5% per-cell false-positive probability x 135k cells floods a Bernoulli decode). Returns
    ``(n, CHANNELS, STEPS)`` {0,1}.
    """
    shape = shape if shape is not None else (n, model.channels, STEPS)
    abar = sched["abar"].to(device)
    steps = sched["steps"]
    context = anchor[0].to(device) if anchor is not None else None
    x = torch.zeros(shape, device=device)
    if context is not None:
        x = torch.maximum(x, context)
    for t in reversed(range(steps)):
        tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
        p0 = torch.sigmoid(model(x, tt))
        if threshold is not None:
            x0_hat = (p0 > threshold).float()
        else:
            draw = torch.rand(shape, device=device, generator=generator)
            x0_hat = (draw < p0).float()
        if context is not None:
            x0_hat = torch.maximum(x0_hat, context)
        if t > 0:
            keep = abar[t - 1]
            u = torch.rand(shape, device=device, generator=generator)
            x = x0_hat * (u < keep).float()
            if context is not None:
                x = torch.maximum(x, context)   # keep context observable to the model
        else:
            x = x0_hat
    if not hard:
        return x
    return (x > 0.5).float()


def discrete_reconstructor(model, sched, device="cpu", generator=None, threshold=None):
    """``predict_fn(context01, mask) -> pred01`` via reveal-from-silence with context anchored."""
    def predict(context01, mask):
        ctx = torch.from_numpy(context01.astype(np.float32))[None].to(device)
        out = sample(model, sched, n=1, anchor=(ctx,), device=device, generator=generator,
                     threshold=threshold)
        return out[0].cpu().numpy()
    return predict


def discrete_sampler(model, sched, device="cpu", generator=None, threshold=None):
    def draw():
        out = sample(model, sched, n=1, device=device, generator=generator, threshold=threshold)
        return out[0].cpu().numpy()
    return draw


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())
