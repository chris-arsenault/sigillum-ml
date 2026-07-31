# Fractal harmonic refinement — measured report

*Contract: `experiment.json`. Runs are regeneratable; generated reports,
checkpoints, and refined examples live under ignored `outputs/`. This file is the
durable, checked-in measured result.*

## Question

Can **one shared learned refinement primitive** expand a coarse functional-harmony
skeleton into a finer per-bar progression — while holding the coarse pillars
fixed — better than copy/n-gram baselines at the coarse steps where genuine
harmonic motion must be invented?

This is the first empirical instantiation of the fractalization direction: a
score attribute lives on a maskable multi-resolution grid, and the same operator
weights refine it at every stride (`16→8→4→2→1`). It deliberately does **not**
encode a fixed `form → harmony → melody → orchestration` pipeline; the grid
semantics are explicit and the inserted content is learned.

## Data and provenance

- Source: Ruby-Partitura-projected `harmonic_function` annotations over the S3
  corpus (`annotation_semantics_v1` + `pilot_v1` manifests, digest-pinned in
  `experiment.json`). Python never parses MusicXML here.
- Four annotated lineages, four movements each: Beethoven, Dvořák, Mozart,
  Tchaikovsky. Vocabulary: 80 key-relative Roman functional tokens plus four
  control tokens.
- Tokens are **key-relative** (`I`, `V`, `V/V`), not absolute (`C:I`). An early
  absolute encoding fragmented the vocabulary and failed to transfer; key-relative
  functions share a vocabulary across composers.
- Inversions / figured-bass digits are collapsed to their function
  (`V7→V`, `ii65→ii`, `I64→I`); they belong to a finer ladder level.

## Two distinct generalization tests (do not conflate)

- **`movement_holdout`** — every composer appears in train; one movement per
  composer is held out for validation and one for test (seen composer, unseen
  movement). Reported over seeds 7, 13, 21.
- **`composer_holdout`** — frozen dataset splits: train Mozart/Tchaikovsky,
  validate Dvořák, test Beethoven (cross composer). Reported over seed 20260731.

Both at 55 epochs, `d_model=160`, 3 layers, 4 heads, window 48 bars, stride 8.

## Results

### Per-step exact-match accuracy — seen composer (`movement_holdout`, test, 3 seeds)

| child stride | learned | copy | unigram | bigram | learned − best baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | **0.355** | 0.320 | 0.305 | 0.318 | **+0.035** |
| 4 | **0.412** | 0.401 | 0.333 | 0.409 | **+0.003** |
| 2 | 0.399 | 0.426 | 0.303 | 0.407 | −0.027 |
| 1 | 0.389 | 0.480 | 0.299 | 0.438 | −0.092 |

Validation mirrors this: learned leads at strides 8 (+0.006), 4 (+0.017), and
ties at 2 (+0.001), then loses at stride 1 (−0.052).

**Interpretation.** The learned primitive beats every baseline exactly at the
coarse steps (`16→8`, `8→4`) where it must invent real harmonic motion. Copy
overtakes it at the finest strides because adjacent bars usually keep the same
chord, so copying the nearest pillar is a strong, almost unbeatable baseline
there. This is the first evidence the shared refinement operator carries useful
information — but only at coarse resolution.

### Recursive full generation and held-out likelihood

| metric | movement_holdout (test) | composer_holdout (test) |
| --- | ---: | ---: |
| recursive full-refinement accuracy (learned) | 0.336 | 0.297 |
| held-out NLL (learned) | 2.426 | 2.409 |
| held-out NLL (bigram LM) | **2.162** | **2.216** |
| authentic adjacent-bar repeat rate | 0.458 | 0.409 |
| learned recursive repeat rate | 0.920 | 0.892 |

**Interpretation (negative).** Under full recursive argmax generation the operator
degenerates toward repeated chords (repeat rate ≈0.9 vs authentic ≈0.45) and its
held-out likelihood still loses to a smoothed bigram LM. It is **not** yet a
usable harmonic generator.

### Cross-composer (`composer_holdout`)

Negative / data-limited. On Beethoven (test) the learned operator only ties/edges
the baselines at the coarsest stride (+0.011 at stride 8) and loses everywhere
else; on Dvořák (validation) it loses at every stride. With only two training
composers, the shared operator does not generalize to an unseen composer better
than an n-gram. This is expected given four annotated lineages and is reported,
not hidden.

## Honest conclusions

1. **Positive, narrow:** a single shared operator produces genuine coarse
   harmonic motion above copy/n-gram baselines at the two coarsest refinement
   steps, on seen composers, replicated across three seeds. This supports the
   fractalization primitive at coarse resolution.
2. **Negative, important:** full recursive generation degenerates to chord
   repetition and loses held-out likelihood to a bigram LM; cross-composer
   transfer is not yet demonstrated.
3. This is one harmonic ladder level only. It is **not** a full-score composer,
   a critic, a selection policy, or an RL reward, and must not be cited as one.

## Reproduce

```bash
python -u -m generation.tools.train_fractal_harmony \
  --split-mode movement_holdout --seeds 7 13 21 --epochs 55
python -u -m generation.tools.train_fractal_harmony \
  --split-mode composer_holdout --seeds 20260731 --epochs 55
```

Outputs (ignored): `report_<mode>.json`, `refined_examples_<mode>.json`, and a
first-seed `checkpoint_<mode>.pt`. The refined-examples JSON is the human-facing
per-bar `authentic → model_refined` artifact; refined slots carry only real chord
functions (control tokens are blocked from decoding), while preserved parent
pillars keep their exact authentic chord — including rare out-of-vocab chords
shown as `<rare>`.
