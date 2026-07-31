# Fractalized whole-score generation (generation.fractal_score)

This package is the first concrete test of the project's core direction: a
successively refined, whole score produced by a learnable coarse-to-fine
process, not an authored form/harmony/melody pipeline and not a review platform.

## The idea

A score attribute is laid out on a maskable grid at the finest resolution. A
coarse representation reveals only a strided subset of the slots (the
"pillars"); a finer representation reveals a denser subset. One shared learned
operator engraves the revealed-parent context into the newly revealed child
slots. Applying that same operator repeatedly, halving the stride each time, is
the fractalization step: a coarse skeleton is engraved into progressively finer
detail by the same primitive, and the coarse skeleton is held fixed (parent
preservation) as detail is added.

The schedule only ever says "make the grid twice as dense." What goes in each
slot, and whether a refinement is good, is learned. Nothing here encodes a fixed
deterministic composition order.

- ladder.py     RefinementSchedule + which grid slots are revealed/refined (general)
- harmony.py    the harmony instantiation: per-bar functional-Roman progressions
- vocab.py      key-relative chord-token vocabulary
- dataset.py    windowing, masked-refinement arrays, seen-composer resplit
- model.py      the shared conditional-Transformer refinement operator
- train.py      masked coarse-to-fine training loop (random level per example)
- baselines.py  copy-nearest-pillar, unigram, bigram, bigram LM
- refine.py     recursive inference, exact-match / NLL / motion-realism evaluation

## First instantiated level: functional harmony

The harmony level uses the Ruby-projected harmonic_function annotations from the
S3 symbolic symphonies (Mozart 41, Beethoven 9, Tchaikovsky 6, Dvorak 9). Each
measure's downbeat carries a key-relative functional Roman numeral (V7 collapses
to V, ii65 to ii); inversions and figured-bass detail are a finer ladder level
for later. Python does not parse any score here: the chords, keys, and offsets
all come from Partitura observations.

Run the experiment (checkpoints/report/artifacts stay under ignored outputs/):

    python -m generation.tools.train_fractal_harmony --split-mode movement_holdout
    python -m generation.tools.train_fractal_harmony --split-mode composer_holdout --seeds 20260731

The measured result is durable in
experiments/whole_score/fractal_harmony_v1/report.md.

## What this is and is not

It is a learned, recursive, shared refinement operator that produces a real
refined harmonic scaffold for held-out movements, measured against mandatory
baselines. It is not a full-score composer, a critic, a selection policy, or an
RL reward, and it deliberately models one attribute at one ladder level. Report
negative results honestly: copy-nearest-pillar is a strong baseline wherever
harmony persists.
