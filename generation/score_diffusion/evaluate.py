"""Evaluation harness: does the denoiser actually reconstruct held-out score structure?

The task is **masked reconstruction**. We blank a contiguous block of measures across all
instrument families in a window, hand the model the surrounding context, and ask it to fill the
hole. We score only the masked cells, on the active (onset) class, because the roll is ~99.8%
empty -- accuracy is meaningless, so we report **active-cell precision / recall / F1**.

Three honesty checks, all required before claiming the generator works:

1. **Beat the baselines** (silence / marginal / persistence) on masked-region active-cell F1.
2. **Generalize to held-out lineages** -- the same margin on the ``test`` split (unseen composers),
   not just ``validation``.
3. **Do not collapse** -- unconditional samples must match the authentic density and repetition
   band, i.e. not decay to silence and not stamp out one repeated column.
"""
from __future__ import annotations

import numpy as np
import torch

from generation.score_diffusion import baselines as B
from generation.score_diffusion.dataset import from_x0, to_x0
from generation.score_diffusion.model import sample
from generation.score_diffusion.roll import CHANNELS, SUBDIV, STEPS


def make_block_mask(mask_measures=2, steps=STEPS, start_step=None) -> np.ndarray:
    """``(CHANNELS, steps)`` mask with ``1`` on a contiguous block of ``mask_measures`` measures
    (all channels). Defaults to a centered block."""
    width = mask_measures * SUBDIV
    width = min(width, steps)
    if start_step is None:
        start_step = (steps - width) // 2
    start_step = max(0, min(start_step, steps - width))
    mask = np.zeros((CHANNELS, steps), dtype=np.float32)
    mask[:, start_step:start_step + width] = 1.0
    return mask


def active_cell_prf(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    """Precision/recall/F1 on active cells within ``mask`` (returns raw tp/fp/fn for micro-averaging)."""
    sel = mask > 0
    p = (pred > 0.5) & sel
    t = (truth > 0.5) & sel
    tp = int(np.sum(p & t))
    fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _micro(rows: list) -> dict:
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "n_windows": len(rows)}


def model_reconstructor(model, sched, device="cpu", generator=None):
    """Return ``predict_fn(context01, mask) -> pred01`` using RePaint anchored sampling."""
    def predict(context01, mask):
        x0 = torch.from_numpy(to_x0(context01))[None].to(device)          # (1,C,STEPS)
        known = torch.from_numpy((1.0 - mask).astype(np.float32))[None].to(device)
        out = sample(model, sched, n=1, anchor=(known, x0), device=device, generator=generator)
        return from_x0(out[0].cpu().numpy())
    return predict


def evaluate_reconstruction(dataset, predict_fn, *, mask_measures=2, limit=None, mask_start=None) -> dict:
    """Micro-averaged masked-region active-cell PRF for one predictor over a dataset split."""
    n = len(dataset) if limit is None else min(limit, len(dataset))
    mask = make_block_mask(mask_measures, dataset.steps, start_step=mask_start)
    rows = []
    for i in range(n):
        truth = dataset.roll01(i)
        context = truth * (1.0 - mask)
        pred = predict_fn(context, mask)
        rows.append(active_cell_prf(pred, truth, mask))
    return _micro(rows)


def repetition_rate(roll01: np.ndarray) -> float:
    """Fraction of non-empty step columns that exactly duplicate the previous non-empty column.

    A collapsed sample (one chord stamped everywhere) approaches 1.0; varied music stays low.
    """
    cols = roll01 > 0.5
    prev = None
    dup = 0
    total = 0
    for s in range(cols.shape[1]):
        col = cols[:, s]
        if not col.any():
            continue
        total += 1
        if prev is not None and np.array_equal(col, prev):
            dup += 1
        prev = col
    return dup / total if total else 0.0


def anti_collapse_report(dataset, sampler, *, n_samples=8, limit_authentic=None) -> dict:
    """Compare unconditional samples to authentic windows on density and repetition.

    ``sampler`` returns a ``{0,1}`` roll for each of ``n_samples`` draws.
    """
    n_auth = len(dataset) if limit_authentic is None else min(limit_authentic, len(dataset))
    auth_density = [float((dataset.roll01(i) > 0.5).mean()) for i in range(n_auth)]
    auth_rep = [repetition_rate(dataset.roll01(i)) for i in range(n_auth)]
    gen = [sampler() for _ in range(n_samples)]
    gen_density = [float((g > 0.5).mean()) for g in gen]
    gen_rep = [repetition_rate(g) for g in gen]
    return {
        "authentic": {"density_mean": float(np.mean(auth_density)),
                      "repetition_mean": float(np.mean(auth_rep)), "n": n_auth},
        "generated": {"density_mean": float(np.mean(gen_density)),
                      "repetition_mean": float(np.mean(gen_rep)), "n": n_samples},
        "collapsed_to_silence": float(np.mean(gen_density)) < 0.2 * float(np.mean(auth_density)),
        "collapsed_to_repetition": float(np.mean(gen_rep)) > float(np.mean(auth_rep)) + 0.25,
    }


def unconditional_sampler(model, sched, device="cpu", generator=None):
    def draw():
        out = sample(model, sched, n=1, device=device, generator=generator)
        return from_x0(out[0].cpu().numpy())
    return draw


def compare_to_baselines(dataset, model, sched, *, mask_measures=2, limit=None, device="cpu",
                         generator=None) -> dict:
    """Full comparison table: model vs each baseline on the same masked windows + anti-collapse."""
    target_density = dataset.coverage()["mean_density"]
    marginals = dataset.channel_marginals()
    results = {}
    for bl in B.default_baselines(marginals, target_density):
        results[bl.name] = evaluate_reconstruction(
            dataset, bl.predict, mask_measures=mask_measures, limit=limit)
    results["model"] = evaluate_reconstruction(
        dataset, model_reconstructor(model, sched, device=device, generator=generator),
        mask_measures=mask_measures, limit=limit)
    results["anti_collapse"] = anti_collapse_report(
        dataset, unconditional_sampler(model, sched, device=device, generator=generator),
        limit_authentic=limit)
    results["beats_all_baselines"] = all(
        results["model"]["f1"] > results[name]["f1"] for name in ("silence", "marginal", "persistence"))
    return results
