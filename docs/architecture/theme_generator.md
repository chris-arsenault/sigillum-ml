# Kernel-Constrained Theme Generator: architecture & decisions

The durable architecture record for growing `generation/theme_gen/`
into a full implementation of the Stage 0–3 spec
(`.sulion-paste/paste-2026-06-13_20-10-32-402Z.txt`, "Kernel-Constrained
Diversity-Injecting Theme Candidate Generator"). The execution plan lives at
`docs/project/THEME-GENERATOR-PLAN.md`; this file records *why* the shape is what it is.

## What the system is (and is not)

It produces a **spread of dissimilar draft candidates** that all satisfy a theme's fixed
obligations (the *kernel*), to break a composer out of convergent self-similar output
while preserving the theme's required identity. Success is the *spread* within the
obligation envelope, not the merit of any one candidate.

**Inviolable non-goal (spec):** the system never judges musicality, never produces a
"best" candidate, never converges toward an exemplar. Any global quality target
reintroduces the convergence the system exists to defeat. Per-candidate randomized
targets stay randomized. Final musical judgment is fully external (human audition).

## Pipeline (Stage 0–3)

- **Stage 0 — kernel:** pinned events (pitch-rhythm, harmonic, structural) + a soft
  `role` bias. Pins are inviolable identity; role only shapes *roll bounds*, never pins.
- **Stage 1 — constrained generation:** a Markov model fills connective material
  *between pins*, conditioned to land each next pin cleanly. Candidates that cannot
  reach a pin are backtracked or discarded. Over-generate; later stages filter.
- **Stage 2 — randomized target conformance over free space only:** per candidate, roll
  independent targets for contour / surprise / self-similarity, measured **only on
  unpinned regions** (pins subtracted), and score distance to the rolled target.
- **Stage 3 — spread selection:** greedily select the batch to maximize mutual
  dissimilarity across the same feature space. Ordering is dissimilarity, not merit.

## Decisions

Recorded here (the repo's durable home for trade-offs; the dated summary is in
`docs/logs/ledger.md`). Each names the rejected alternatives and the consequence.

### D1 — Corpus: pluggable ingestion + music21 bundled default
**Decision.** A corpus interface with two sources: (a) a MIDI/MusicXML **ingestion path**
reading from `assets/raw/corpus/`, and (b) **music21's bundled public-domain corpus**
(e.g. Bach chorales, folk) as the default general-tonal training set. Repo-local locked
themes remain a selectable corpus. The trained model is **persisted** as an artifact.
**Alternatives rejected.** music21-only (cannot train on supplied datasets later);
ingestion-only (no licensing-safe default, every run needs hand-supplied data).
**Consequences.** Adds an ingestion tool and a model-artifact format/loader. Licensing is
contained to whatever a user drops into `assets/raw/corpus/`; the default ships clean.

### D2 — Generator: hybrid DP feasibility gate + bounded backtracking
**Decision.** Precompute a **reachability/feasibility table** (which pitch+duration states
can still reach the next pin within the remaining beats under all active constraints),
weighted-sample only among feasible moves, and drop to **bounded backtracking** only when
sampling stalls in a tightly-pinned region.
**Alternatives rejected.** Pure backtracking CSP (combinatorial blow-up on long free
spans); pure guided sampling (paints into corners, cannot *prove* a clean landing).
**Consequences.** The feasibility table is the load-bearing new primitive; it also feeds
density reporting (D5). Backtracking needs an explicit step/time bound to stay bounded.

### D3 — Model: context-conditioned, persisted
**Decision.** Condition transitions on **phrase position, meter accent, and harmonic
context**, support higher/variable order, train once, and persist the artifact loaded by
the generator. This is what makes the surprise proxy and self-similarity meaningful.
**Alternatives rejected.** Modest first-order + accent only (surprise stays shallow);
swap-corpus-only (gaps 2 & 8 stay open).
**Consequences.** Larger training/serialization surface. Surprise remains explicitly
**soft and optional** per spec — it must be droppable without affecting the other two
dimensions.

### D4 — Kernel authoring extends the item-list DSL (no JSON/YAML)
**Decision.** Kernels are authored in the repo's existing **item-list notation DSL**
(`(pitch | [pitches] | None, quarterLength, *tokens)` + the `score.py` helpers), via a
thin kernel-authoring helper layer and small Python kernel modules. A
`generation/tools/gen_themes.py` CLI loads a kernel by id/path and emits the batch + report.
**Alternatives rejected.** JSON schema, YAML schema (both duplicate a notation language
the repo already has, and fight the established compose-in-Python → audition → lock
workflow).
**Consequences.** "Generate batches without editing Python" means *without editing the
generator/library code* — you write a small DSL kernel file and run the CLI. No new
dependency. In-progress kernels live in `experiments/`; locked ones in
`symphony/materials/`.

### D5 — Density / free-space / expected-spread reporting
**Decision.** Per kernel, compute and surface **kernel density, free-space percentage,
and an expected-spread bound** (the tighter the kernel, the narrower the achievable
spread — a true property, surfaced, not hidden). Reuses D2's feasibility machinery.

## Authoring workflow (D4)

Kernels are authored in the item-list DSL — terse `frame` / `pin` / `harm` / `phrase` /
`kernel` constructors exported from `generation.theme_gen` (no JSON/YAML). A kernel
module exposes `KERNEL` (a `ThemeKernel`); `generation/tools/gen_themes.py` loads it by dotted module
path or `.py` file, runs the batch, and writes the audition score + spread report:

    python -m generation.tools.gen_themes experiments.s_beloved_kernel --seed 20260613

In-progress kernels live in `experiments/`; locked ones in `symphony/materials/`. Compose a
kernel → generate → audition → lock, mirroring the repo's melody workflow. No
generator/library edits are needed to author or run a new theme.

## Module structure

The prototype graduates from a single 1k-line module into a package
`generation/theme_gen/` (corpus / model / engine / features / density / kernel-DSL
/ report) plus an internal `_common` for frame-agnostic primitives. The public import
surface is `generation.theme_gen`. Verify gate: `python -m unittest discover -s tests`.
