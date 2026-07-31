# fractal_denoise_v2 — discrete (Bernoulli) whole-score denoiser (phase 11)

## Why v2

`fractal_denoise_v1` showed a continuous Gaussian DDPM is the wrong prior for a 99.8%-empty
*binary* onset roll: it decoded ~200x too dense and lost to the persistence baseline. v2
reformulates corruption/generation as **discrete Bernoulli** — corrupt by surviving active onsets
to silence, generate by revealing onsets from silence with class-balanced occupancy prediction —
reusing the same roll, lineage splits, baselines, and honesty gates.

## What was built (durable)

- `generation/score_diffusion/discrete.py` — cosine survival schedule, `q_sample` (survive-to-
  silence corruption), `OccupancyDenoiser` (per-cell occupancy logits), class-balanced BCE loss,
  and a **predict-x0 ancestral sampler** (predict occupancy, re-corrupt to the next level) with
  RePaint context anchoring and optional threshold decode.
- `train.py::train_discrete` + CLI `generation/tools/train_score_denoiser.py` (CPU-capped <=3
  threads, `--decode-threshold`), contract `experiment.json`, and 6 discrete tests (26 total).

## Smoke run (NOT the contract run)

CPU-bounded: 2 threads, width 96, depth 6, 60 diffusion steps, 8 epochs, `pos_weight=1`, 8-16 eval
windows/split. The contract is 40 epochs / width 192 / depth 8 / 200 steps / full split.

| split | persistence F1 (bar) | discrete model F1 | gen density (auth) |
| ----- | -------------------- | ----------------- | ------------------ |
| validation | 0.185 | 0.007 (Bernoulli) / 0.003 (thr 0.5) | 0.37 (0.0023) |
| test | 0.109 | 0.005 / 0.004 | 0.37 (0.0018) |

## Findings (honest)

1. The discrete training signal is real and much healthier than continuous. BCE falls
   0.53 -> 0.089; on in-distribution corrupted inputs the model separates classes well:
   `p0~0.75` on true onsets vs `p0~0.05` on empty cells at low noise, and `p0~0.008` (about the
   corpus marginal) on a fully-silent input. So occupancy *is* learnable from the raw surface.
2. Decoding the ultra-sparse grid still floods, so it does not yet beat persistence. A 5%
   per-cell false-positive probability x ~135k cells swamps the ~180 true onsets. A Bernoulli
   decode floods (density 0.37); a fixed threshold either floods (0.5) or reveals nothing (>=0.7).
3. Masked infilling is out-of-distribution for an unconditional model. With the masked region
   simply blanked, a single global noise level cannot tell the model "this region is to be filled"
   vs "genuinely silent," so `p0` inflates in the hole. The anchored ladder fills but over-produces.

## Recommended next step

The blocker is not the diffusion formulation but (a) decoding precision on an ultra-sparse grid and
(b) OOD infilling. Two concrete, principled levers, in priority order:

1. Condition on an observed-context channel. Add a per-cell "observed" indicator to the model
   input so infilling is in-distribution (the model knows which silence is context vs target). This
   is the standard conditional-inpainting fix and should sharply cut the masked-region flood.
2. Factor the output to shrink the false-positive surface. Instead of 135k independent Bernoulli
   cells, predict, per `(family, step)`, a small active-pitch set (occupancy + top-k pitch logits).
   This matches the true sparsity structure (few pitches sound per family per step).
3. Then run the full-capacity contract (40 epochs, width 192) — infeasible interactively on a
   2-3-thread CPU cap; it needs a longer, capped background run or a GPU.

Persistence on the unseen-lineage `test` split (F1 0.109) remains the bar to clear before any
materialization or audition work. Checkpoints and `report.json` are gitignored under `outputs/`.
