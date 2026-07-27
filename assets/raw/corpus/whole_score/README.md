# Whole-score corpus staging

This directory contains external whole-score research material fetched from the
tracked registry at `corpora/whole_score/sources.json`.

All downloaded archives, extracted sources, integrity state, normalized
Partitura observations, and derived training examples below this directory are
ignored by Git. Only README files are tracked.

Generated layout:

```text
fetch-state.json
sources/
  <source-id>/
    downloads/
    source/
```

Do not make Python training code interpret MusicXML, Humdrum, MuseData, or MIDI
as a second score framework. A later ingestion stage will route score semantics
through Partitura Ruby and retain the raw source digest in every normalized
observation.
