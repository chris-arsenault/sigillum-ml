# Theme-generator training corpus (ingestion input)

Drop MIDI / MusicXML melodies here to extend the general-tonal Markov training corpus
(decision D1, `docs/architecture/17_theme_generator.md`). The ingestion tool reads this
directory; the default training set is music21's bundled public-domain corpus, so this
directory may be empty.

**Licensing:** you own the licensing of anything added here. Keep only material you are
entitled to train on. Generated model artifacts and outputs live under `outputs/` (ignored).

## Per-style subfolders and separate models

Files directly in this folder are picked up by the `ingest` source. Subfolders let a
distinct style train its own model via `ingest:<subfolder>`:

    assets/raw/corpus/arabian/   ->  --sources ingest:arabian   (separate maqam model)
    assets/raw/corpus/<style>/   ->  --sources ingest:<style>

Modern material (e.g. video-game themes, melodic-trance leads) is not bundled with
music21 and is licensing-bound, so it is **not** auto-sourced; drop your own MIDI here to
include it. The default `music21` source ships clean (Bach + Essen folksong).
