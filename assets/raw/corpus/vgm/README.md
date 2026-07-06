# vgm corpus

Drop vgm melody MIDI / MusicXML here. Build a model from it:

    # separate model (keeps vgm intervals out of the tonal default):
    python -m generation.tools.build_corpus_model --sources ingest:vgm --out vgm.json

    # or blend it into the general model by moving files up one level into
    # assets/raw/corpus/ (root), then rebuilding the default model.

You own the licensing of anything added here (see ../README.md).
