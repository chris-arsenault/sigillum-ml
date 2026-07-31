# Real-candidate seam plausibility v1

Date: 2026-07-29

## Question

On real Partitura composition-workflow continuations, does the frozen
structural-context V4 signal track a musician's ear where the fixed
boundary-profile baseline cannot decide?

This is a small plausibility probe, not a trained critic benchmark and not a
review-platform milestone. It is designed to produce a genuinely non-trivial
human decision, not a self-graded accuracy number.

## Real score material

The three eight-measure studies are reductions of current Movement IV material
from:

`../sigillum/symphony/movements/mvt4_merged/dsl/mvt4_storyteller_merged.rb`

Each carries the source flute, clarinet counterline, and harp color anchor
without generated note expansion:

- **plain telling:** Movement IV bars 9-16;
- **slow narrator:** Movement IV bars 41-48;
- **driving return:** Movement IV bars 61-68.

The first four source measures are the fixed anchor. For each open continuation
the real Ruby workflow receives four candidates:

1. `coherent_a`: the authentic next four Movement IV measures;
2. `hard_alternative`: a musically plausible, boundary-matched alternate
   continuation in the same register and accompaniment;
3. `coherent_b`: a second hand-written plausible continuation;
4. `discontinuous`: a gross key/register/continuity break, kept only as a
   positive control.

Ruby Partitura scheduled `span:continuation`, applied each candidate as a real
source patch, and compiled, snapshotted, and normally exported all 12
candidates. Python scored only the canonical observations returned for those
successful exports.

## Why the review pair is worth your ears

The human batch compares `coherent_a` (the composer's authentic continuation)
against `hard_alternative`. That alternative shares the anchor's final-bar
boundary exactly, so the **immediate boundary baseline is tied on all three
items and cannot express a preference**. Any preference therefore comes only
from the learned four-measure residual. There is no obvious "wrong" option and
no answer key the model was fit to; the decision is a real musical judgment.

## Model-only result (no human data yet)

Review pair — authentic `coherent_a` versus `hard_alternative`:

| Method | Prefers authentic | Prefers alternative | Ties |
| --- | ---: | ---: | ---: |
| Boundary baseline | 0 | 0 | 3 |
| V4 learned mean | 1 | 2 | 0 |

So the boundary heuristic is undecided on every item, and V4 takes a definite
position on every item (authentic in the slow narrator; the alternative in the
plain telling and driving return). Whether those positions match a human is the
open question this batch exists to answer.

Positive control — authentic versus gross discontinuity: V4 `3/3`, boundary
`3/3`. Both reject gross breakage, so it is excluded from the human batch to
avoid wasting review time.

## Judgment

There is no plausibility verdict yet: the learned signal is decisive exactly
where the baseline is blind, which is what makes the human comparison
informative, not what settles it. V4 remains unsuitable as an automatic
selector, calibrated critic, controller reward, or RL signal until the human
preferences below are collected and analyzed against these frozen scores, with
the boundary baseline retained as the mandatory comparison and held-out lineage
transfer required before any policy use.

## Populated human check

The same workflow run builds three blinded A/B comparisons (authentic versus
plausible alternative), one per excerpt. Every bundle contains the normal
Partitura MusicXML and MIDI; the WAV previews are real General MIDI renders of
those MIDI files (flute, clarinet, harp) through FluidSynth, peak-normalized so
A and B sit at the same level.

Reproduce the scoring and rebuild the populated generic review manifest:

```bash
python -m generation.tools.run_seam_plausibility --write
```

Review it through the shared cadence-agnostic tool:

```bash
cd tools/sigillum-review-ui
npm install
npm run dev
```

The tool reads
`outputs/reviews/whole_score/seam_plausibility_v1/review-manifest.json`, serves
the referenced WAVs directly, and writes choices to
`outputs/reviews/whole_score/seam_plausibility_v1/review-results.json`.

Real General MIDI rendering needs FluidSynth and a GM soundfont. Provision them
once with `nix build --no-link nixpkgs#fluidsynth nixpkgs#soundfont-fluid`, or
point `SIGILLUM_FLUIDSYNTH` and `SIGILLUM_REVIEW_SOUNDFONT` at your own copies;
the runner refuses to emit silent or sine-only audio.

Generated media and results remain outside Git. The durable review content is
the real source-derived note material, all explicit alternatives, the generic
manifest contract, tool, experiment contract, runner, and this measured result.
