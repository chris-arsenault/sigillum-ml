# Whole-score generation: what we are trying, what we learned, and what comes next

This document explains the whole-score generation work in ordinary language. It is deliberately
more tutorial than API reference. The goal is that you should be able to understand **what each
experiment is testing**, **what its result means**, and **which decisions are technical decisions
the implementation should own** rather than unexplained choices pushed back to you.

## The short version

We want a model that can create a complete orchestral score by repeatedly refining an initially
unstructured score-shaped object:

```text
noise / almost-empty score
    -> broad section and phrase shape
    -> harmonic and rhythmic structure
    -> notes and instrumental activity
    -> orchestration, voice-leading, articulation, and dynamics
    -> complete Partitura score
```

This is inspired by [diffusion models](https://en.wikipedia.org/wiki/Diffusion_model), which learn
to reverse a corruption process, and by
[multiscale modeling](https://en.wikipedia.org/wiki/Multiscale_modeling), which represents one
system at several levels of detail.

We call this **fractalized generation**, but that is our project term rather than a standard
machine-learning method. We do **not** mean that a musical score is literally a mathematical
fractal. We mean:

1. represent the score at several scales;
2. corrupt or simplify a real score;
3. train a learned operator to reconstruct the missing detail;
4. apply refinement repeatedly, from coarse structure toward fine detail;
5. use the same general learning framework at more than one scale instead of encoding a rigid,
   handwritten composition recipe.

The important current result is:

> The full corpus contains enough information to learn a real onset-reconstruction signal, but our
> present representation and decoder produce far too many false notes. The immediate blocker is
> not a lack of scores and not the review UI. It is how we represent and decode an extremely sparse
> orchestral note surface.

## What the model currently sees

The canonical score and all musical materialization remain in Ruby Partitura. Python does not parse
MusicXML or become a second score runtime. It receives facts already projected by Partitura:

- measure boundaries;
- part and instrument names;
- note onsets;
- MIDI pitches;
- durations and score positions (although the current roll does not yet model all of them).

We convert those facts into a representation inspired by a
[piano roll](https://en.wikipedia.org/wiki/Piano_roll), a common representation in
[music information retrieval](https://en.wikipedia.org/wiki/Music_information_retrieval):

```text
rows    = 12 instrument families x 88 MIDI pitches = 1,056 channels
columns = 8 measures x 16 positions per measure    = 128 time steps
cell    = 1 if that family/pitch has a note onset there, otherwise 0
```

Relevant implementation:

- [`roll.py`](roll.py) — score observation -> onset roll and active-cell round-trip;
- [`dataset.py`](dataset.py) — windows over the 110-score corpus;
- [whole-score workflow description](../../docs/research/whole_score_composition_workflows.md).

### Why normalize every measure to 16 positions?

A downbeat should line up with other downbeats even when pieces use different meters. We therefore
map every measure to 16 relative positions. This is a form of
[feature engineering](https://en.wikipedia.org/wiki/Feature_engineering): we preserve structural
position inside the measure rather than absolute clock time.

The tradeoff is that a step in 3/4 does not represent the same quarter-note duration as a step in
4/4. This is acceptable for the first structural test but will eventually need a richer duration
and meter representation.

### What is missing from the current roll?

The current roll is **not yet a complete score representation**. It records note onsets, coarse
instrument family, pitch, and relative position. It does not fully represent:

- note duration and ties;
- individual parts within a family;
- voices and contrapuntal identity;
- dynamics, articulation, phrasing, and expression;
- explicit harmonic function;
- section/phrase boundaries;
- long-range movement structure.

That is intentional for an early experiment: first establish whether the model can learn a real
note-surface distribution. But success on this roll would still be a prerequisite, not the final
composer.

## The basic learning idea

The experiments use [self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning):
the corpus supplies its own teaching signal.

1. Take a real score window.
2. Corrupt, simplify, or hide part of it.
3. Ask the model to reconstruct the original.
4. Compare the reconstruction with the known original.

No human has to label every note as good or bad. This resembles a
[denoising autoencoder](https://en.wikipedia.org/wiki/Denoising_autoencoder), except that diffusion
models learn a sequence of corruption levels and reverse steps instead of one fixed corruption.

For filling a missing passage, the related visual analogy is
[inpainting](https://en.wikipedia.org/wiki/Image_inpainting): preserve known context and generate
the missing region.

## What we have tried

### 1. Deterministic composition workflow and learned critics

Before the current generator experiments, the project built the surrounding workflow:

- execute candidate score changes safely in Ruby Partitura;
- preserve trajectories, promotion, and rollback;
- compare candidates;
- learn structural seam and pairwise preference signals;
- build audition/review tooling.

These pieces are useful infrastructure, but they do **not** by themselves generate a score. A
classifier or critic answers "which candidate looks better?" It does not answer "where do the
candidates come from?" We therefore moved critic/controller/RL work behind the generator.

Related general ideas:

- [binary classification](https://en.wikipedia.org/wiki/Binary_classification);
- [representation learning](https://en.wikipedia.org/wiki/Feature_learning);
- [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) — deliberately
  deferred until there is a generator and a validated reward;
- [human-in-the-loop](https://en.wikipedia.org/wiki/Human-in-the-loop) evaluation — useful after
  the model produces meaningful alternatives.

Relevant project records:

- [`structural_context_v4`](../../experiments/whole_score/structural_context_v4/);
- [`critic_v1`](../../experiments/whole_score/critic_v1/);
- [`seam_plausibility_v1`](../../experiments/whole_score/seam_plausibility_v1/).

**What it taught us:** learned signals can detect some structural differences, but critics are not
a substitute for a generative model.

### 2. Coarse-to-fine harmonic refinement (`fractal_harmony_v1`)

We first tested the fractal idea on chord/harmony tokens because a chord sequence is much smaller
than a full orchestral score. The model was given a coarse set of harmonic "pillars" and repeatedly
revealed the missing chords between them.

Conceptually this combines:

- a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain)-like sequence of refinement states;
- [conditional probability](https://en.wikipedia.org/wiki/Conditional_probability), because new
  chords are predicted given the revealed pillars and context;
- a discrete masking/reconstruction task;
- [language modeling](https://en.wikipedia.org/wiki/Language_model)-style baselines such as
  unigram and bigram chord models.

Result:

- it beat simple baselines at the two coarsest refinement steps on seen composers;
- recursive generation collapsed toward repeated chords;
- it lost held-out generative likelihood to a bigram model;
- cross-composer generalization was weak;
- the experiment only had explicit chord-function annotations for 16 movements / 4 lineages.

This did **not** prove the whole-score idea works. It did show that a shared refinement operator can
learn something at a limited scale, and that recursive generation and distribution collapse are
real risks.

Read: [`fractal_harmony_v1/report.md`](../../experiments/whole_score/fractal_harmony_v1/report.md).

### 3. Continuous Gaussian score diffusion (`fractal_denoise_v1`, phase 10)

The next experiment used all 110 scores without chord labels. We treated the onset roll like an
image and used a continuous
[denoising diffusion probabilistic model](https://en.wikipedia.org/wiki/Diffusion_model):

1. map empty/active cells to continuous values;
2. add [Gaussian noise](https://en.wikipedia.org/wiki/Normal_distribution) at a random noise level;
3. train a convolutional network to predict/remove that noise;
4. generate from random Gaussian noise by repeatedly denoising;
5. preserve known context with a RePaint-style inpainting anchor.

The network uses:

- a [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) over
  time;
- [residual connections](https://en.wikipedia.org/wiki/Residual_neural_network);
- dilated convolutions to see a wider time span without an enormous network;
- a cosine noise schedule so the final corruption state is genuinely close to pure noise.

Relevant code: [`model.py`](model.py), [`train.py`](train.py).

Result:

- the complete train/sample/evaluate pipeline worked;
- training loss decreased;
- the model generated about **42% active cells** where authentic scores contain about **0.2%**;
- masked reconstruction F1 was roughly 0.005 versus roughly 0.185 for the persistence baseline.

**Why it failed:** Gaussian image diffusion assumes a continuous field. Our current score surface is
an extremely sparse set of yes/no events. Background cells dominate the loss, while even a small
false-positive probability creates thousands of invented notes.

Read: [`fractal_denoise_v1/report.md`](../../experiments/whole_score/fractal_denoise_v1/report.md).

### 4. Discrete/Bernoulli diffusion (`fractal_denoise_v2`, phase 11)

Because each cell asks a yes/no question — "is there an onset here?" — we replaced Gaussian noise
with a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution).

Forward corruption:

```text
real onset -> survives with probability p(t), otherwise becomes silent
empty cell -> remains empty
```

Reverse generation:

```text
start mostly/all silent
predict the probability that every cell was active in the clean score
reveal likely onsets
repeat at successively finer corruption levels
```

The model is trained with
[binary cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy), the standard loss for predicting
yes/no probabilities. We also tested class weighting because active onsets are rare.

Relevant code: [`discrete.py`](discrete.py),
[`train_score_denoiser.py`](../tools/train_score_denoiser.py).

Result:

- the training signal is substantially healthier than continuous diffusion;
- BCE fell from about 0.53 to 0.089 in the CPU smoke run;
- on true onset cells the model produced probabilities around 0.75;
- on empty cells it produced probabilities around 0.05;
- but 5% false-positive probability across about 135,000 cells still produces a huge flood;
- generated density remained about 0.37 instead of 0.002;
- it still did not beat the persistence baseline.

This is an example of
[class imbalance](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis):
99.8% of cells are negative. A classifier that looks numerically "pretty good" per cell can still be
musically catastrophic because there are so many opportunities for false notes.

Read: [`fractal_denoise_v2/report.md`](../../experiments/whole_score/fractal_denoise_v2/report.md).

## Why we use these evaluation rules

### Lineage-level train/validation/test splits

The 110 scores are grouped into 24 musical lineages. Entire lineages are assigned to train,
validation, or test. We do not randomly scatter neighboring movements/windows across splits.

This guards against [data leakage](https://en.wikipedia.org/wiki/Leakage_(machine_learning)): a
model should not receive one movement or near-duplicate in training and then be credited for
recognizing its sibling in test.

### Why accuracy is useless here

If 99.8% of cells are empty, a model that predicts "empty" everywhere gets about 99.8% accuracy
while composing nothing.

We therefore report:

- [precision](https://en.wikipedia.org/wiki/Precision_and_recall): of the notes predicted, how many
  were correct?
- recall: of the real notes, how many were recovered?
- [F1](https://en.wikipedia.org/wiki/F-score): a combined precision/recall score;
- generated density: did it collapse to silence or flood the score?
- repetition: did it stamp the same chord or column repeatedly?

### Mandatory baselines

A learned model matters only if it beats simple methods:

| Baseline | Meaning | Why it matters |
| --- | --- | --- |
| Silence | Predict no notes | Exposes misleading accuracy on sparse data |
| Marginal frequency | Predict generally common instrument/pitch cells | Tests whether the model learned more than global popularity |
| Persistence/time-copy | Copy the preceding musical block | Strong because real music repeats locally |
| Unigram/bigram harmony | Predict common chord or chord transition | Strong small-data baseline for chord sequences |

The persistence baseline is currently the bar to beat: about **0.185 validation F1** and **0.109
test F1** in the smoke harness.

## What the current failure actually means

It does **not** mean:

- "we do not have enough symphonic data" — phase 10 and 11 use all 110 projected scores;
- "diffusion cannot work for music" — the discrete model learns a real separation signal;
- "we need a better review website" — no review tool can repair a model producing hundreds of
  times too many notes;
- "we should move to reinforcement learning" — RL would optimize an inadequate generator and an
  unvalidated reward;
- "we need to hand-code the composition process" — that would abandon the learnable framework.

It means:

1. the independent-cell representation creates about 135,000 chances to make a false-positive
   mistake per window;
2. the current unconditional model is not explicitly told which cells are observed context and
   which cells are intentionally masked for reconstruction;
3. a small per-cell probability error becomes an enormous musical error;
4. the next architecture must model the **structure of sparsity**, not merely reweight the loss.

## The next methods, explained without jargon

### A. Add an observed-context channel — do this next

During infilling, a zero can mean either:

- "we observed this location and it is truly silent," or
- "we hid this location and want the model to generate it."

The current model cannot distinguish those cases. We should give it an extra binary mask saying
which region is observed. This turns the task into explicit conditional
[inpainting](https://en.wikipedia.org/wiki/Image_inpainting) rather than hoping an unconditional
model infers our intent.

Expected effect: reduce out-of-distribution behavior in the missing region and improve precision.

This is a technical implementation decision, not something you should need to choose.

### B. Factor activity from pitch — likely immediately after A

Today the model independently asks 88 pitch questions for every instrument family at every time
step. That produces too many chances for false positives.

A better factorization is:

```text
Question 1: is this instrument family active at this time?       (mostly no)
Question 2: if active, which one/few pitches are present?        (small choice)
```

This is related to a [hurdle model](https://en.wikipedia.org/wiki/Hurdle_model) or
[zero-inflated model](https://en.wikipedia.org/wiki/Zero-inflated_model): first model whether an
event exists, then model its value conditional on existence.

Expected effect: shrink the false-positive surface from 135,000 independent yes/no decisions to a
much smaller number of activity decisions plus conditional pitch choices.

### C. Learn a lower-dimensional latent score space — later, if A+B are insufficient

[Latent diffusion](https://en.wikipedia.org/wiki/Latent_diffusion_model) does not denoise raw image
pixels. It first compresses them into a smaller learned representation and diffuses there.

For scores, a latent representation could capture chords, texture, instrumental density, rhythmic
activity, and voice-leading without exposing every possible note cell independently.

Expected effect: easier global modeling and fewer false positives. Cost: another learned encoder and
decoder whose reconstruction quality must be proven before the generator can be trusted.

### D. Add harmony/chord conditioning — auxiliary, not the core generator

We can train on sourced chord progressions or derive harmonic features from the score corpus. Those
features can condition generation, similar to text conditioning in image diffusion.

Useful for:

- controlling harmonic direction;
- testing whether explicit harmony reduces note-level ambiguity;
- pretraining a harmony representation;
- keeping long passages coherent.

But chord conditioning alone does not solve rhythm, melody, counterpoint, orchestration, or form.
It should guide the general score model rather than replace it.

### E. Expand to the actual multiscale score hierarchy — after the note surface works

The north-star hierarchy remains:

```text
movement length/form
  -> sections
    -> phrases
      -> measures
        -> beats/rhythm
          -> notes/voices
            -> orchestration/expression
```

The present 8-measure onset model only tests one lower layer. Once it can reconstruct/generate
plausible note surfaces, we can add coarse structural variables and train refinement between these
levels. Building the full hierarchy before the basic generator clears its baselines would create
more machinery around an unproven core.

## Who should decide what

### Technical decisions the implementation should own

You should **not** be asked to choose, without evidence, between:

- Gaussian DDPM vs Bernoulli/discrete diffusion;
- BCE weight values;
- probability threshold vs stochastic sampling;
- convolution width/depth;
- observed-mask conditioning;
- factored output parameterization;
- training schedules and ablations.

Those choices should be made by implementing the smallest honest comparison, running it against the
same held-out baselines, and following the measured result.

### Product/musical decisions that genuinely need your input

Your input matters for questions such as:

- What counts as a useful first output: 8 measures, 30 seconds, or a complete movement?
- Should the first target sound idiomatically symphonic, stylistically novel, or controllable by a
  reference/harmonic plan?
- Which audible failures are unacceptable even when numerical metrics improve?
- How much training time or compute is worth spending once a smoke experiment has a real positive
  signal?
- Which generated alternatives are musically preferable once the model produces plausible choices?

Until there are plausible alternatives to hear, model architecture and evaluation are engineering
responsibilities, not review-platform decisions.

## Current recommendation

The technically justified sequence is:

1. add explicit observed/hidden conditioning and train the infilling task in-distribution;
2. factor family activity from pitch choice to reduce false positives;
3. run a small CPU-capped smoke comparison against the same persistence baseline;
4. only if the smoke result improves materially, run the full-capacity training contract;
5. then materialize samples through Ruby Partitura and listen;
6. add explicit harmony and higher-level structure as conditioning/refinement levels;
7. defer critics, preference learning, controller policies, and RL until the generator offers
   plausible alternatives.

This is not asking you to choose between A and B. A should be implemented first because it fixes a
known mismatch between training and evaluation. B should follow because it fixes the known
false-positive geometry of the representation.

## Deeper references

Accessible overviews above are mostly Wikipedia. The experiments are also directly related to:

- Ho et al., [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — the
  continuous phase-10 starting point;
- Nichol & Dhariwal,
  [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) — cosine
  noise schedule;
- Austin et al.,
  [Structured Denoising Diffusion Models in Discrete State-Spaces](https://arxiv.org/abs/2107.03006)
  — the discrete-state motivation behind phase 11;
- Lugmayr et al., [RePaint](https://arxiv.org/abs/2201.09865) — conditioning diffusion by repeatedly
  re-imposing known context during inpainting.

## Reproducing the experiments

All training commands run in the foreground and are CPU-capped to protect the development host.
Generated checkpoints/reports stay under gitignored `outputs/`.

```bash
# Continuous Gaussian experiment (phase 10)
python -m generation.tools.train_score_diffusion --threads 2

# Discrete/Bernoulli experiment (phase 11)
python -m generation.tools.train_score_denoiser --threads 2

# Tests
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -m pytest -ra -o addopts=
```

The CLIs reject more than 3 Torch threads on this six-core host.
