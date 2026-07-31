# fractal_denoise_v1 — full-corpus self-supervised score denoiser (phase 10)

## Goal

First end-to-end test, on the **whole** 110-score corpus with **no labels**, of the
"start from noise, refine toward a plausible full score" direction. Concretely: train a
continuous DDPM denoiser over a meter-normalized multi-family onset roll and check whether it
can (1) reconstruct masked measures of held-out scores better than silence / marginal /
persistence baselines, on both the `validation` and unseen-lineage `test` splits, and (2)
generate unconditionally without collapsing to silence or one repeated column.

This generalizes the earlier label-scarce `fractal_harmony_v1` probe: same fractal
coarse-to-fine idea, but over the full corpus and the raw note surface instead of 16 annotated
harmony movements.

## What was built (durable)

- `generation/score_diffusion/roll.py` — meter-normalized onset roll: 12 instrument families ×
  88 pitches = **1056 channels**, 8 measures × 16 subdivisions = **128 steps**; keyword family
  classifier; exact active-cell round-trip to structured events (materialization stays in Ruby
  Partitura).
- `dataset.py` — sparse window store reusing the manifest **lineage split** (train 79 / val 16 /
  test 15 scores → 2384 / 665 / 932 windows); `[-1,1]` transforms; channel marginals; coverage
  incl. fallback-part counts.
- `model.py` — self-contained continuous DDPM (dilated Conv1d + FiLM), **cosine** noise schedule
  (near-pure-noise terminal at any step count), sparse-active auxiliary loss, ancestral sampling
  with optional RePaint coarse/masked anchor.
- `baselines.py` — silence, calibrated marginal-frequency, persistence (time-copy).
- `evaluate.py` — masked-measure reconstruction, active-cell precision/recall/F1, held-out
  lineage split, anti-collapse density/repetition.
- `tools/train_score_diffusion.py` — foreground train+eval CLI, **capped at ≤3 CPU threads**.
- Contract `experiment.json`; hermetic tests `tests/test_score_diffusion.py` (20 tests).

## Smoke run (NOT the contract run)

A small CPU-bounded run to exercise the whole path (2 threads, width 96, depth 6, 60 diffusion
steps, 6 epochs, 8 eval windows/split). This is **not** the `experiment.json` contract
(40 epochs, width 192, depth 8, 200 steps, full split). Train loss fell 1.25 → 1.16.

| split | baseline silence F1 | marginal F1 | persistence F1 | model F1 | gen density (auth) |
| ----- | ------------------- | ----------- | -------------- | -------- | ------------------ |
| validation | 0.000 | 0.053 | **0.185** | 0.005 | 0.42 (0.0023) |
| test | 0.000 | 0.022 | **0.109** | 0.004 | 0.42 (0.0018) |

## Honest finding

**Negative so far.** The denoiser does **not** beat the baselines; persistence (time-copy) is far
ahead. The failure mode is diagnostic, not just under-training: the sampled field decodes ~**0.42
active density vs ~0.002 authentic** — ~200× too dense — so precision is ~0.002 while recall is
high. A continuous Gaussian DDPM over a 99.8%-empty binary roll, thresholded at 0, spends its
capacity modelling the empty background and its sampled variance floods the grid with false
onsets. The cosine-schedule and sparse-active-loss fixes reduced but did not remove this.

This is consistent with the `fractal_harmony_v1` conclusion that the mask-reveal ladder is really
a **discrete** diffusion process: the onset roll is Bernoulli, not Gaussian.

## Recommended next step (for phase 11)

Reformulate the corruption/denoising as **discrete/Bernoulli** over onset occupancy rather than
continuous Gaussian: predict per-cell onset logits with a class-balanced BCE (or a D3PM-style
absorbing-state schedule), and decode by a calibrated per-window occupancy target instead of a
fixed 0 threshold. Keep the same roll, splits, baselines, and honesty gates; the persistence F1
above is the bar to clear on the held-out `test` split before any materialization/audition work.

Generated checkpoints and `report.json` live under gitignored `outputs/`; this report, the
contract, the code, and the tests are the durable record.
