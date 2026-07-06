# Detailer (fractalizer) — experiment registry

## STATUS: PAUSED (2026-06-19) — resume here

The monophonic detailer **works and is usable** (v2). On a clean hand-written theme (RENN) it adds
in-key (98%), density-controllable, velocity-shaped surface detail — the first NN output in the
project that is both usable and doing the job the tool is actually good at (detailing, not composing).
Paused while the composer considers other approaches.

**Resume commands** (all CPU; artifacts are gitignored, the model rebuilds from the corpus):
- Build dataset (only if the corpus/manifest changed — now carries velocity):
  `python -m generation.tools.build_theme_dataset` (parallel, ~4 min)
- Train the detailer: `python -m generation.tools.train_detailer --steps 8000` (~40 min) → `outputs/models/detailer/`
- Detail / eval: `python -m generation.tools.detail --density 1.5` → `outputs/sketches/detailer/{detailed,renn_v2}.mid`
  (lower `--density` = lighter ornament; the RENN demo transposes E♭→C, details, →back)
- Metrics: MLflow sqlite, `experiment detailer`, runs `detailer_v1` / `detailer_v2`.

**Open threads, priority order:**
1. **Onset-only anchoring** — full-cell anchoring is buggy (reproduces the input); pin only the
   structural notes' *onset* cells, leave the rest free, for clean preservation (>83%).
2. **v3 — figuration-map conditioning** (ControlNet-style, built from the figuration extractor) →
   discrete, on-demand ornaments (run/arp/turn requested per region).
3. **v3 — multi-voice / orchestral** (octave doublings + inner voices) — the real 07b under-elaboration
   pain; monophonic is the warm-up.
See the full proposed-changes set and the v1→v2 results below.

---

The pivot from neural theme *composition* (see `docs/research/neural_theme_generation.md`) to neural
**detailing**: the model's real strength is adding surface density to a given line, not inventing the
tune. A conditional diffusion model takes a coarse melodic line and elaborates it toward the corpus's
detail distribution — chosen because diffusion **samples the detailed tail** instead of regressing to
the bland mean (the failure that capped the autoregressive model). Conceptually image img2img /
super-resolution on a piano-roll. Metrics in MLflow (`mlflow.db`, experiment `detailer`).

| Version | Setup | Outcome |
| --- | --- | --- |
| detailer_v1 | DDPM, 1-D dilated-conv FiLM denoiser (1.37M); roll = 1/12-grid pitch+rest+onset image, 16-beat windows; condition = coarse **beat-skeleton** roll (concat); 26K corpus windows | **premise validated, not yet usable.** val 1.07→0.052. Adds **varied** detail incl. **triplets** (16ths/8th-trip/16th-trip/8ths) — NO eighth-collapse. 83% structural-note preservation. Reconstruction fixed (onset-authoritative + modal pitch → no per-cell wobble fragmenting held notes). **But on the real RENN theme: over-busy (~280n vs 85), only 78% in-key (no key awareness), and beat-skeleton coarsening reinvents rather than enriches.** |
| detailer_v2 | v1 + **velocity** channel (4th, continuous), **key-normalised** to C, **ornament-strip** coarsening, and conditioning on **density + character + harmony with classifier-free guidance** (1.39M) | **usable.** val→0.047. On RENN: **98% in-key** (was 78% — key-norm fixed the chromatic clash); **density is dial-able** (94 notes @ target 1.5, 121 @ 3.5 — vs v1's fixed ~280 torrent); **velocity present + shaped** (was flat); light, theme-appropriate detailing. **Anchoring bug**: full-cell RePaint re-imposes the coarse → output==input (no detail); off, the coarse-roll *conditioning* preserves ~83%. Next: **onset-only anchoring** (pin structural onsets, free the rest) for clean preservation; figuration-map + multi-voice (v3). |

## The full proposed-changes set (cumulative — these COMPOSE, none replaces another)

These accumulated across discussion; the v2 build folds them into ONE dataset rebuild + ONE retrain,
with the denoiser's conditioning interface designed so every signal is just-another-input (so v3
items slot in without re-architecting).

### v2 — make the monophonic detailer usable on real lines
**Representation (roll channels + key):**
- **Velocity channel** — a 4th, *continuous* roll channel (loudness heatmap); model learns dynamic
  shaping (accents, swells, cadential lean). Capture velocity in `events.py` (currently dropped at
  parse); coarse velocity is flattened so the model *adds* the shaping.
- **Key-normalize** — transpose every training window to C; at inference transpose theme→C, detail,
  →back. Ornaments land in-key by construction (fixes the 78%-in-key drift).

**Coarsening (how training pairs are built):**
- **Ornament-strip coarsening** — figuration extractor removes passing/neighbour/run/trill notes,
  keeps the structural line (replaces the beat-skeleton). Teaches *enrich*, not *reinvent-from-skeleton*.

**Conditioning (compose into one interface, all with classifier-free guidance):**
- **Density / detail-strength scalar** — dial light↔florid (corpus ~3.4 notes/beat is too busy for a theme).
- **Character + harmony/key embeddings** — the "toward a goal" steer; CFG (train with cond-dropout,
  amplify at inference) so "more heroic / denser / more 16ths" actually pushes.

**Sampling:**
- **Inpaint-anchor** the given notes (re-impose known cells each reverse step) — never altered, only decorated.

### v3 — richer control + expressivity
- **Figuration-map conditioning (ControlNet-style)** — a per-region figuration image (run / arp /
  turn / grace) built from the figuration extractor; model renders the requested ornament where asked.
  Makes figurations *discrete and on-demand*, not just emergent. (Stretch: a VQ codebook of figuration
  patches → discrete composable units.)
- **Multi-voice / orchestral** — octave doublings + inner voices (multi-channel/stacked rolls). This
  is the actual 07b under-elaboration pain (block half-notes, no doublings); monophonic is the warm-up.

## Standing notes
- The established hand-written themes (`symphony/materials/themes.py`: RENN, ESHAIA, ONSLAUGHT,
  MEMORY, STORYTELLER, SAELITH) are already detailed/finished — they are the *target* level, not the
  coarse input. The real use case is detailing the **plain agent-written accompaniment / inner voices**
  (the 07b under-elaboration failure: block half-notes, no 16ths, no doublings).
- Monophonic first; orchestral (octave doublings, inner voices) is multi-voice diffusion, later.
- Code: `generation/fractal/{roll,dataset,model}.py`, `tools/{train_detailer,detail}.py`.
