# Model Card — ThemeGPT v0 (single-token baseline)

- **Status:** trained, evaluated, superseded by v1
- **Artifacts:** `s3://sigillum-mlops-559098897826/theme_nn/v0_single_token/` (model.pt, vocab.json, metrics.csv, sample.mid, model_card.md)
- **Code:** `symphony/materials/theme_nn/` @ commit of "generation + decode" (5551272)

## Hypothesis
A small decoder-only Transformer over **one-token-per-event** sequences, conditioned on
key + shape + character, can learn to generate coherent in-key melodies from the curated corpus —
the baseline the deferred motif-developer must beat.

## Intended use
Research baseline for kernel-constrained neural theme generation. Generates free melodies
conditioned on key / contour-shape / character tags. **Not** kernel-constrained (no pins/infilling).

## Data
- **1,588 examples / 490,018 events** from the curated, categorized corpus (vgm + classical +
  exotic + arabian; rejects excluded).
- Melody = **`analysis.skyline`** (naive global top-line — weaker than `theme_gen/corpus.py`'s
  voice-scorer used by the Markov; a known v0 deficiency).
- Pitch encoded **key-relative** (scale degree + alteration + octave); durations quantized to 1/12.
- Control signals: per-bar harmony (carried, **not yet conditioned on**), shape archetype,
  character tags (from the categorization manifest), key.

## Architecture & training
- ThemeGPT (nanoGPT-shaped, `torch.nn`): 4 layers, d_model 256, 4 heads, block 384, dropout 0.1.
- **Vocab 4,170** (single compound token per event) → **5.4M params** (embedding/head dominated).
- AdamW lr 3e-4, wd 0.01, batch 24, 700 steps, CPU (~2.2 s/step). Loss on the event region only.

## Metrics
| step | train | val |
| ---- | ----- | --- |
| 1 | 8.37 | 8.09 |
| 200 | 4.26 | 4.28 |
| 400 | 3.81 | 3.99 |
| 700 | 3.23 | **3.91** |

Val **plateaued ~3.9 (perplexity ≈ 50) by step 400** while train kept falling → data/capacity
limited, mild overfit after.

## Evaluation (qualitative — five M1-character samples, decoded and inspected)
- **Mode collapse at decode.** temp 0.7 → a single note repeated ~80× (`C5`); temp 0.9 → hammers
  `F5` (126/154); temp 1.1 → scatters across 4 octaves. Not melodies.
- **Character conditioning works:** heroic sits high (avg ~76), romance drops low (~66),
  **battle+tension pulls in chromaticism** (70% in-key with raised tones vs ~97% diatonic else).
- **Shape conditioning does not work:** per-third average pitch is flat in every sample ("arch" ignored).
- **Data is not the cause:** training data is only **9% repeated-note** — a model emitting ~99%
  repeats has collapsed. Cause = small/undertrained model + single-token data-inefficiency.

## Limitations
Undertrained & data-limited; single-token vocab (4,170) is data-inefficient (every degree×oct×dur
combo learned independently); mode-collapses under sampling; no harmony conditioning; no
pins/infilling; weak melody extractor.

## Outcome → next
Keep GPT. The single-token representation wastes the data. **v1: factor the event** (degree /
alteration / octave / duration as separate small fields with their own embeddings + heads) so the
model shares musical primitives → learns from less data, ~10× fewer embedding params. Also: switch
to the `corpus.py` melody extractor; add octave augmentation. No decode hacks.
