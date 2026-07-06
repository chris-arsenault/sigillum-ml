# Model Card — ThemeGPT v1 (factored)

- **Status:** trained, evaluated
- **Artifacts:** `s3://sigillum-mlops-559098897826/theme_nn/v1_factored/` (model.pt, vocab.json, metrics.csv, sample.mid, model_card.md)
- **Code:** `generation/theme_nn/{vocab_factored,model_factored}.py`, `generation/tools/train_theme_nn_factored.py` @ commit aaf846a
- **MLflow run:** experiment `theme_nn`, run `v1_factored`

## Hypothesis
Factoring the event into separate small fields (kind / degree / alteration / octave / duration),
each with its own embedding and prediction head, makes the GPT **share musical primitives** across
notes → it learns from less data with fewer parameters than v0's single 4,170-word token.

## What changed from v0
- Vocabulary: one compound 4,170-token per event → **~92 field-tokens total** (durations binned to
  a 12-value musical palette; degree 8, octave 11, alt 3, cond 52, kind 6).
- Parameters: **3.3M** (vs v0 5.4M) — the embedding/head shrank ~50×.
- Same transformer backbone (4 layers, d_model 256, 4 heads, block 384), same data (1,588 examples).

## Metrics (per-event NLL, comparable to v0's CE)
| step | v0 val | v1 val |
| ---- | ------ | ------ |
| 100 | 4.77 | 4.40 |
| 400 | 3.99 | 4.04 |
| 700 | **3.91 (plateau)** | 3.98 |
| 1500 | — | **3.82 (still improving)** |

v1 final field NLLs: kind 0.01, **degree 1.46** (the bottleneck — melodic pitch choice), alt 0.34,
oct 0.80, dur 1.00. v1 reaches a lower NLL with fewer params and **had not plateaued** at 1,500
(v0 flattened by 400) → less data-starved.

## Evaluation (five M1-character samples, decoded and inspected)
**Improved over v0:**
- **Much less mode-collapse.** v0 t0.7 = `C5`×80 (pure repeat); v1 t0.7 moves (`C4 G4 C4 D4 F4 Eb4 C5…`),
  and t0.9 / t1.1 drop to **8–9% repeated-note** (v0 hammered F5). Factoring broke the attractor.
- Character conditioning persists: battle+tension is chromatic (71% in-key, busy 16ths) vs heroic diatonic/high.

**New artifacts introduced:**
- **Octave outliers** — octave is predicted *independently* of degree, so notes occasionally land
  5 octaves out (ranges span midi 0–94, e.g. `Db1`, `Fb1`). Unmusical.
- **Choppy rhythm** — stray `0.083` rests scattered through.
- **Early EOS** — `romance+aria` emitted only 5 notes; the kind/EOS head fires too early for some conditioning.

## Outcome → next (v2 levers)
Hypothesis **supported**: factoring is more parameter- and data-efficient and less collapse-prone.
But it traded single-note collapse for octave-scatter + rhythm noise. Next:
1. **Constrain the octave field** — clamp to a sane melodic range, or predict octave *conditioned on
   degree/previous pitch* (kill the independence that lets a note jump 5 octaves).
2. **Fix EOS calibration** (early stop) and the stray micro-rests.
3. **Switch the dataset to `theme_gen/corpus.py`'s voice-scoring melody extractor** (v0/v1 used the
   weaker naive skyline — the choppiness partly comes from there).
4. Then more steps / data. Not the give-up point — the curve and samples both moved the right way.
No decode hacks.
