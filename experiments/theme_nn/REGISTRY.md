# ThemeGPT — experiment registry

The experiment loop: **version → hypothesis → config → outcome → notes → repeat.** Each run has a
model card here and its artifacts (checkpoint, vocab, metrics, sample output, analysis) in S3 at
`s3://sigillum-mlops-559098897826/theme_nn/<version>/`. Metrics are tracked in MLflow (`mlruns/`).

The goal: a kernel-constrained neural theme generator that beats the Markov baseline (coherent but
boring) — without hand-coded generation heuristics. We get the model right or we give up.

| Version | Hypothesis (1-line) | Val ppl | Outcome |
| ------- | ------------------- | ------- | ------- |
| [v0_single_token](v0_single_token/model_card.md) | a small GPT over single-token-per-event sequences learns coherent melodies | 50 | **mode-collapses** at decode; data is fine (9% repeat), model is data-inefficient + undertrained |
| [v1_factored](v1_factored/model_card.md) | factoring the event (degree/oct/dur as separate fields) shares statistics → learns from less data, fewer params | **46** | **supported** — 3.3M params (vs 5.4M), still improving at 1500 (v0 plateaued@400), much less collapse. New artifacts: **octave outliers** (octave predicted independently → notes jump 5 octaves), choppy rhythm, early EOS → v2 |
| infill_etude (dual) | T5 span-infill + full feature stack; predicts **absolute degree/oct** and re-ranks pitch by 1st/2nd/3rd-order interval heads at generation | core-val ~4.0 | **dead end** — interval-order gating produced real runs/arcs (fixed 1st-order random-walk oscillation), but the absolute-degree head **locks onto a repeated pitch** ~half the seeds, and **more training made it WORSE** (3/6 musical @3k → 0/6 @8k): lower per-token loss = sharper head = stronger repeat attractor (classic LM degeneration). Structural ceiling of predicting absolute pitch. |
| infill_ivlprimary | **interval-primary**: predict the next INTERVAL (the move), pitch = running sum from the left pin, last note clamped to land the next pin; degree/oct now input-only context (what the coherent Markov does) | core-val 3.65 | **lock fixed, new trade-off** — **11/12** M1 fills musical (vs 3/6 dual), 16-31 distinct pitches, no lock; traversal-to-pin makes sitting-on-a-note impossible. BUT only **69% in-key** (dual was 98%): summing *semitone* intervals has no key anchor → chromatic drift. Next: diatonic. |
| infill_diatonic | **diatonic-interval-primary**: predict the move as a DIATONIC STEP (scale position = octave*7+degree, so pitch lands on a scale tone by construction) + a separate per-note ALTERATION (accidental) | core-val 3.85 | **both properties achieved** — **12/12** M1 fills musical (no lock), **98% in-key** (vs ivl-primary 69%), 21-31 distinct pitches, scalar runs/arcs, chromaticism only as occasional deliberate accidentals (alt-head learned alt=0 dominant). Synthesis of interval-primary's anti-lock + the dual model's key-safety. Best result. Pending: listening verdict vs Markov. |

## Conclusion (2026-06-18) — where this landed, and why we're pivoting

**The arc.** v0/v1 (free-running GPT) mode-collapsed. The infilling kernel model + full feature stack
got the *machinery* right (pins land, spans tile, output in-key, leaps controlled), and the
representation search ended at **diatonic-interval-primary** — the best output by every metric: 12/12
M1 fills "musical", 98% in-key, no pitch-lock, scalar runs/arcs, chromaticism only as deliberate
accidentals. It beats the Markov on coherence.

**But it is not good music, and it cannot get there from here.** Listening verdict: *"in the right
direction but not anywhere close enough."* The remaining deficits are all **structural** — no motif
(every note invented fresh), no phrase/arc, and rhythmic monotony (the duration head collapses the
corpus's 16ths/8ths/triplets into ~80% eighths-and-sixteenths, dropping triplets entirely).

**One root cause: per-token cross-entropy → regression to the mean.** The same force surfaced three
times — the absolute-degree head locked onto the tonic (*more* training made it WORSE: 0/6 musical at
8k vs 3/6 at 3k), the duration head collapsed to eighths, and the contour random-walked. Minimising
next-token loss rewards the locally-safe modal choice, which is exactly what makes the output
competent and characterless. Structure is what a small local-likelihood model erases.

**Diagnostics that bounded it.** (1) *Model-side, not kernel-side* — the lock follows the model to
whatever pitch it lands on, independent of the pins. (2) *Fills to density, doesn't ornament* —
pinning every beat just doubles the notes; the model connects pins to corpus density, it does not
decorate a skeleton. (3) *Corpus is rich, output is flat* — data has 1026 distinct durations,
generation uses 14.

**Realistic ceiling.** ≈3.3M params on ~4.1K melodies under per-token likelihood tops out at
"competent but generic". Reference: folk-RNN needed ~23K tunes for *coherent* (still generic) melody;
good models use 100K–1M+ MIDI — we are ~5× short of basic coherence by data alone, and more data
mostly lowers loss, the very thing *causing* the collapse. "A lot better" needs both far more data
*and* a different objective/process (hierarchical/structured generation, preference tuning), not a
tuning pass. The model is best read as an **ornamenter/assistant, not a theme-composer**. See the
write-up: `docs/research/neural_theme_generation.md`.

## Backlog (candidate next experiments)
- **Metric awareness — the model must know beat position and bar index.** The event stream is
  currently metre-agnostic (durations only, anchored to the first note), so a downbeat is
  indistinguishable from, say, a 7th-16th-note offbeat, and bar 3 from bar 2 — yet those are
  musically different and govern harmony/figuration. Encode each note's beat-in-bar (and bar index
  / phrase position) into the representation + conditioning so the model can place a note in the
  metre. (Discussed at length; was dropped in the rhythm-relative representation.)
- **Anacrusis / pickup handling** — honor a melody that starts before bar 1 beat 1, **using only
  the MIDI file's own bar/time-signature information** (time-sig events, bar markers). If the file
  doesn't carry enough metric information, do NOT try to detect the downbeat from the raw notes —
  fall back to the onset-0 assumption. Pairs with the metric-awareness item.
- **Shrink the backbone** — now that factoring cut the vocab cost to 0.03M, the 3.16M transformer
  blocks are the only cost. Try `d_model 192, 3 layers` (~1.3M total, same factored rep) and read
  params-vs-loss. (Unlocked by v1; wasn't meaningful before — the 4170 vocab dominated.)
- ~~Condition octave on degree / clamp range~~ — **addressed** by interval-gated generation (the
  interval head vetoes a wild octave leap when it reranks the pitch); revisit only if outliers persist.
- **Encode per-bar harmony into the conditioning** — carried in every example, not yet fed to the model.
- **Fix EOS calibration + stray micro-rests** (v1 early-stop on some conditioning).
- **Make `shape` actually bind** (contour ignored through v1).

## Done
- **Voice-scoring melody extractor** — `analysis.skyline` (crude global top-note) deleted and
  replaced by `analysis.melody_line`: splits the texture by voice, scores each for melody-likeness
  (register + pitch-class variety − time-based polyphony, ported from `theme_gen/corpus.py`), and
  returns the winning voice's top line as real NoteEvents. Added a span-coverage gate so a sparse
  cue/sting can't out-score the lead. Takes effect on the next dataset rebuild.
- **Metric inputs + interval-gated pitch + étude heads** — the infill representation feeds **beat +
  bar** (metre, from the file's time signature, anacrusis-aware) as input fields. The decoder predicts
  absolute degree/octave for pitch, plus four étude heads: figuration, motif, chord-position (loss
  weight 0.3), and **melodic interval (weight 1.0)**. Interval is load-bearing: at generation it
  **reranks the joint degree×octave choice** (voice leading steers absolute pitch, and a wild octave
  leap gets vetoed). Wired + mechanically verified (the gate overrides a forced far-octave leap; loss
  trains all heads; tiling holds); not yet trained. Dataset carries every per-event feature.

## Standing notes
- **Data is not the bottleneck** (measured: 9% repeated-note rate, healthy duration spread). v0/v1
  used the weaker `analysis.skyline`; the extractor is now voice-scored (`analysis.melody_line`,
  ported from the Markov's `theme_gen/corpus.py`) — pending a dataset rebuild to take effect.
- **No decode hacks** (repetition penalty / nucleus) — that is just hand-programming the melody.
  Fixes go in the model/representation/training, or we give up.
- Conditioning that already works: **character** (shifts register + chromaticism). Does not
  work: **shape** (contour ignored) — unaddressed through v1.
- **Factoring's independence cuts both ways:** v1 broke the single-note collapse (good) but lets
  octave be sampled independently of degree (bad — wild octave jumps). v2: condition octave on
  degree / previous pitch, or clamp the octave range.
