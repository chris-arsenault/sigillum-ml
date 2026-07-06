# Neural Theme Generation for Sigillum — a white paper / post-mortem

*Status: research complete for this phase (2026-06-18). Outcome: the machinery works, the music does
not, and we understand why. This documents the approach, what worked, what failed, the root cause,
and a realistic assessment of what it would take to do better.*

---

## 1. Goal

A **kernel-constrained neural theme generator**: given a *kernel* — fixed pinned notes at fixed
metric positions plus a harmonic / shape / character frame — fill the free spans between the pins and
land the pins exactly, producing a melody for a symphony movement. The kernel carries the composer's
intent (where the theme begins, its apex, its harmony, its character); the model supplies the
connective melody. The bar to clear was the existing **Markov** generator: coherent but boring.

The standing rule throughout: **no decode hacks** (repetition penalties, nucleus sampling tricks).
Those are just hand-programming the melody. Fixes had to go into the model, representation, or
training, or the approach gets abandoned.

## 2. The corpus and the feature-extraction subsystem

The training data is the curated multi-part corpus (classical / Bach / VGM), ~4,100 melodies after
cleaning, ~1.03M note events. Two subsystems feed the model:

- **Melody extraction (`framework/analysis/events.py:melody_line`).** Most corpus files are
  multi-part. We split the texture by voice (track×channel / part), score each voice for
  melody-likeness (register + pitch-class variety − *time-based* polyphony, with a span-coverage gate
  so a sparse sting can't out-score the lead), and take the winning voice's top line as real
  NoteEvents. This replaced a crude global-skyline extractor that hopped between voices.
- **Feature tiers (`framework/analysis/`).** Per-note, windowed, fully automatic (no hand labels):
  tonal (scale degree), **harmony** (STFT-like rolling-window chord detection → Roman numerals +
  chord-tone/NCT role, computed from the accompaniment = texture − melody), **figuration**
  (run/arp/trill/neighbour/leap/repeat/sustained + interval-exact sequence), **motif** (developing
  variation: exact / transpose / inversion / tonal-transpose / retrograde / augmentation), and
  **metric** (beat-in-bar + bar index, from the file's real time signature, anacrusis-aware).
- **Categorisation pipeline (`generation/tools/categorize_corpus.py`).** A resumable, idempotent LLM-judgment
  pass tags every piece with a VGM-framed dramatic taxonomy (battle / boss / overworld / romance /
  …, primary + supporting) and moves genre-misfits to a quarantine. The manifest
  (`trackers/corpus_categories.jsonl`) is the single source of truth; newly-unpacked archives become
  pending automatically.

The training dataset (`generation/theme_nn/dataset.py`) pairs each melody (key-relative events) with
its control frame (per-bar harmony, shape archetype, character tags, key) and every per-note feature.

## 3. The representation / architecture journey

| Stage | Representation | Result |
| --- | --- | --- |
| Markov | order-2 interval transitions | coherent but boring; order ceiling |
| v0 | single-token-per-event GPT | mode-collapses at decode |
| v1 | **factored** event (degree/oct/dur as separate embedded fields) | more data-efficient; but octave predicted independently → 5-octave leaps |
| infill | **T5 span-infilling encoder-decoder** over the kernel; étude heads (figuration/motif/chord-pos/interval) | machinery works (pins land, tiles); pitch from absolute degree + interval re-rank |
| dual | absolute degree/oct, re-ranked by 1st/2nd/3rd-order interval heads | fixed the leap oscillation, but **locks onto a repeated pitch**; more training made it worse |
| ivl-primary | predict the **interval** (the move); pitch = running sum, last note clamped to land the next pin | **lock fixed** (12/12 varied) but only **69% in-key** — semitone sums drift chromatic |
| **diatonic** | predict a **diatonic step** (scale position) + a separate **alteration** field | **best**: no lock AND 98% in-key |

## 4. What worked

- **The kernel / infilling machinery.** Spans always tile to the exact beat-length; pins always land
  (the last fill note of a gap is clamped to the next anchor). This is solid and reusable.
- **Diatonic-interval-primary representation.** Predicting the move as a *scale step* (so the
  reconstructed pitch lands on a scale tone by construction) plus a *per-note* accidental gave the
  two properties at once: interval-primary's anti-lock movement (a gap is a pitch traversal from one
  pin to the next, so the model physically cannot sit on a note) **and** the dual model's key-safety
  (98% in-key; chromaticism only where the alteration head deliberately raises a note — and because
  the accidental is per-note, not accumulated, it cannot drift).
- **The feature stack.** Voice-scored melody, real-metre beat/bar inputs, and the étude heads
  (figuration/motif/chord-position predicted to shape the trunk) are all wired and trained.
- **MLOps.** Experiment registry, model cards, MLflow (sqlite-backed) tracking, S3 model store.

By every metric we could compute, the final model beats the Markov: in-key, no lock, controlled
leaps, runs and arcs instead of a random walk.

## 5. What failed — and the one cause behind it

**Listening verdict: "in the right direction but not anywhere close enough."** It is in-key and
locally smooth but **aimless** — and the deficits are all *structural*:

- **No motif.** Every note is invented fresh; there is no recurring idea stated and developed, which
  is what makes a tune memorable rather than merely correct.
- **No phrase / arc.** The pins anchor the endpoints; between them the line wanders without building
  to anything. The `shape`/contour conditioning never bound.
- **Rhythmic monotony.** The corpus has 1,026 distinct durations (24% sixteenths, 20% eighths, ~19%
  triplets, quarters, dotted values); generation uses **14** distinct durations and is ~80%
  eighths-and-sixteenths with **triplets dropped entirely**. A melody's rhythm carries as much
  identity as its pitch; ours has none.

**The single root cause is the training objective: per-token cross-entropy → regression to the
mean.** Minimising the loss of the *next* token rewards the locally-safe, *modal* choice. The same
force produced three independent-looking failures:

1. The absolute-degree (dual) model **locked onto the tonic** on ~half the seeds — and crucially,
   *more* training made it **worse** (0/6 musical at 8k steps vs 3/6 at 3k): lower loss = a sharper,
   more confident head = a stronger repeat attractor. This is the classic degeneration of
   likelihood-trained autoregressive models (text LLMs have it too; their usual fix is the sampling
   tricks we ruled out).
2. The **duration head collapsed** to the modal value (eighth note), erasing the corpus's rhythm.
3. The pitch contour **random-walked** before we added directional structure.

Per-token likelihood, at this scale, optimises for *plausibility*, and the most plausible next note
is the average one. Average notes make competent, characterless music. Structure — motif, phrase,
rhythmic identity — is exactly what averaging destroys.

## 6. Diagnostics that bounded the problem

- **The lock is model-side, not kernel-side.** Filling a non-tonic-saturated kernel produced the same
  lock rate; the model locks onto *whatever* pitch it lands on, not the kernel's tonic. So a richer
  kernel does not fix it.
- **The model fills to density; it does not ornament.** Pinning *every beat* and asking it to
  embellish within each beat just **doubled the notes** — a half-beat gap holds one note (corpus
  density), and pin-landing forces that note to the next beat's pitch. The model connects a skeleton;
  it does not decorate one. (A coarser skeleton, every 2 beats, does get real connective motion — but
  it is the same wandering quality, now shorter.)
- **The corpus is rich; the output is flat.** (§5) The failure is in generation, not the data.

## 7. Realistic assessment — can it get "a lot" better, and how much data?

**Not from this setup.** ≈3.3M parameters, ~4,100 melodies, per-token likelihood — the realistic
ceiling is "competent but generic," roughly where folk-RNN-class models land. We can patch symptoms
(rhythm variety, some motif, shape-binding), each a real effort with bounded gains, but not reach
memorable, characterful tunes.

**Data needed — an estimate by precedent, not a formula.** Two reasons it cannot be a clean
calculation: scaling laws predict *loss*, and we have directly observed loss and musical quality come
*apart* (the eighth-note collapse minimises loss); and there is no quality metric that tracks the
music — the ear is the metric. Anchoring to comparable systems:

- **folk-RNN** produced *coherent* (still generic) monophonic melodies on **~23,000 tunes**. We have
  ~4,100 — roughly **5–6× short of basic coherence** by that precedent.
- Large symbolic-music corpora (Lakh MIDI ≈ 176K files) and models that sound good use **100K–1M+**
  MIDI — ~25–250× our data.

The load-bearing caveat: **more data alone will probably not break the ceiling.** It lowers loss, and
lower loss is what *causes* the modal collapse — so scaling the same objective may just give smoother,
more confident mush. "A lot better" requires **both** much more data **and** a different
objective/process: hierarchical / structured generation (compose a skeleton, then ornament — though
note §6, the model does not ornament well yet), or preference / reward optimisation on a musical
signal, rather than pure next-token likelihood.

## 8. Recommendation

The neural model, at any scale realistically reachable here, is best understood as an
**ornamenter / assistant, not a theme-composer.** For the symphony's *actual* themes the
higher-leverage path is the existing **kernel-constrained framework + texture-card library + human
composition**, with the model assisting (fills, variations, suggestions) rather than generating
themes end-to-end. Pursuing the neural route to "good" is the 40K–100K-tunes + structured-objective
road — a deliberate research program, not a tuning pass.

## 9. Reproducibility

- Code: `generation/theme_nn/` (representation, dataset, infill model, generation),
  `framework/analysis/` (feature tiers), `tools/{build_theme_dataset,train_infill,generate_infill,
  categorize_corpus}.py`.
- Experiment log + model cards: `experiments/theme_nn/REGISTRY.md`.
- Metrics: MLflow, sqlite backend (`mlflow.db`), experiment `theme_nn`.
- Artifacts (dataset, checkpoints) are git-ignored; the dataset rebuilds from the corpus +
  manifest via `python -m generation.tools.build_theme_dataset` (parallel, ~3 min on this box).
