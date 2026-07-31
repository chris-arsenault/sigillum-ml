# Whole-score evaluation lab

This directory contains frozen benchmark definitions, not generated outputs or
claimed results. A benchmark manifest pins each brief and starting Partitura
score by SHA-256, declares all baseline strategies and required ablations, and
defines the complete run and blinded-comparison matrix.

Generated score sources, trajectories, run JSONL, private review mappings,
preferences, MusicXML, MIDI, and reports belong under ignored experiment
storage such as `outputs/evaluation/`; they must not be committed here.

The smoke manifest proves the contract and tooling with one deliberately small
case. It is not a musically meaningful benchmark result. Production studies
must add multiple frozen briefs, multiple seeds, independent raters, and power
analysis before making comparative claims.

Commands:

```bash
python -m generation.tools.evaluate_composition verify \
  experiments/whole_score/v1_smoke/manifest.json

python -m generation.tools.evaluate_composition collect \
  experiments/whole_score/v1_smoke/manifest.json \
  --runs outputs/evaluation/v1_smoke/runs.jsonl \
  --source PATH_TO_COMPLETED_SCORE.rb \
  --case kernel-study --strategy deterministic_graph \
  --seed 1729 --trajectory PATH_TO_TRAJECTORY.jsonl \
  --model-calls 3 --wall-seconds 12.5

python -m generation.tools.evaluate_composition report \
  experiments/whole_score/v1_smoke/manifest.json \
  --runs outputs/evaluation/v1_smoke/runs.jsonl \
  --reviews outputs/evaluation/v1_smoke/reviews.jsonl \
  --preferences outputs/evaluation/v1_smoke/preferences.jsonl
```

Partitura owns score measurement and blinded score rendering. This Python lab
only verifies frozen experiment inputs, stores ML run metadata, joins held-out
human evidence, and aggregates plural diagnostics without producing a
composite reward or automatic winner.

## Learned score-span representation

`representation_v1/experiment.json` freezes the first shared encoder pilot over
the Partitura-bound analytical targets. Run and verify its ignored artifacts
with:

```bash
python -m generation.tools.train_score_span_encoder train
python -m generation.tools.train_score_span_encoder verify
```

The durable measured result is in `representation_v1/report.md`. The pilot is
an analytical representation experiment, not a learned critic. Its test split
has now been evaluated and must not be repeatedly reused for architecture
tuning under an “untouched holdout” claim.

`representation_v2/` corrects the v1 within-target imbalance with uniform-label
sampling. It is validation-only: the exposed v1 test split is not silently
reused for a new generalization claim.

`structural_context_v1/` replaces flat analytical-label prediction with a
bounded self-supervised score task. It learns ordered measure and span
representations by ranking the authentic next four-measure span above a
nonadjacent span from the same score. Its measured report is deliberately
negative: the model learns the task but does not beat immediate boundary
profile distance, so it is not promoted as a critic.

`structural_context_v2/` proves that adding the fixed boundary baseline as a
residual does not help while the negatives remain random; validation selects
the untrained step-zero checkpoint. `structural_context_v3/` controls the
supervision by matching distant same-score negatives on immediate boundary
distance. `structural_context_v4/` combines that harder task with residual
learning and records the first replicated validation improvement plus a
positive external PDMX holdout. The retained artifact is exposed only as a
structural-seam signal, not as a general critic or automatic selection policy.

`seam_review_v1/` turns that narrow signal into one concrete listening check:
six anonymous authentic-versus-splice comparisons, one per external lineage,
with a single A/B/Same review page. Build and verify the ignored audio package
with:

```bash
python -m generation.tools.build_seam_review build
python -m generation.tools.build_seam_review verify
```

The reviewer does not run these commands; they produce the ready-to-open page
under `outputs/reviews/whole_score/seam_review_v1/public/index.html`.

`seam_plausibility_v1/` is the first transfer test on actual composition
workflow candidates rather than historical continuation splices. Three
eight-measure excerpts carry current Movement IV flute, clarinet, and harp
material; the authentic next four measures and three explicit alternatives all
pass the real Ruby compile/snapshot/export path (12/12). The human batch
compares the authentic continuation against a boundary-matched plausible
alternative, where the immediate boundary baseline is tied on all three items,
so the decision isolates V4's learned residual (V4 picks authentic once and the
alternative twice). Gross discontinuities are auto-rejected `3/3` and excluded
from the human batch. This is an open plausibility question pending human ears,
not a positive verdict, and V4 is still unsuitable as a selector, controller
reward, or RL input. Reproduce the scoring and build the three-item
real-content A/B review with:

```bash
python -m generation.tools.run_seam_plausibility --write
```

The durable result is
`seam_plausibility_v1/report.md`; generated Partitura MusicXML/MIDI, MIDI-derived
browser previews, manifests, and preference results remain under ignored
`outputs/`. Review them with the shared local tool:

```bash
cd tools/sigillum-review-ui
npm install
npm run dev
```

The tool serves WAV directly and prints its published Sulion URL.

`critic_v1/` freezes the five criterion-specific pairwise learning gates and
model contract. Audit it with:

```bash
python -m generation.tools.train_pairwise_critics index \
  --source outputs/RUN/trajectory.jsonl \
           outputs/RUN/reviews.jsonl \
           outputs/RUN/preferences.jsonl
python -m generation.tools.train_pairwise_critics audit
```

The audit remains not ready until real trajectory reviews and blind human
preferences are collected under `outputs/`, indexed, split by run, and pinned
by digest. Repeat `--source` when indexing multiple evidence sets. Training
cannot reinterpret analytical annotations or
medium-quality agent decisions as critic rewards.

## Fractalized generation

`fractal_harmony_v1/` is the first experiment on the *generation* side of the
kernel rather than critics/selection. It tests whether one shared learned
refinement operator can engrave a coarse functional-harmony skeleton into a
finer per-bar progression, holding the coarse pillars fixed, across the ladder
`16→8→4→2→1`. The grid/mask semantics are explicit; the harmonic content in the
newly revealed slots is learned; the same operator weights are reused at every
stride, so applying it recursively is the fractalization step. Harmony, key,
measure, and offset facts come from existing Ruby-Partitura observations — Python
does not become a second score runtime. Reproduce with:

```bash
python -u -m generation.tools.train_fractal_harmony \
  --split-mode movement_holdout --seeds 7 13 21 --epochs 55
python -u -m generation.tools.train_fractal_harmony \
  --split-mode composer_holdout --seeds 20260731 --epochs 55
```

The durable result is `fractal_harmony_v1/report.md`. It is deliberately mixed:
the shared operator beats copy/unigram/bigram baselines at the two coarsest
refinement steps on seen composers (replicated over three seeds), but full
recursive generation degenerates toward repeated chords, its held-out likelihood
still loses to a bigram LM, and cross-composer transfer is not yet demonstrated
with only four annotated lineages. This is one harmonic ladder level, not a
full-score composer, critic, selection policy, or RL reward. Generated reports,
`checkpoint_*.pt`, and the per-bar `authentic → model_refined` examples remain
under ignored `outputs/`.
