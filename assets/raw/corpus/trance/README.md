# trance corpus

Drop trance melody MIDI / MusicXML here. Build a model from it:

    # separate model (keeps trance intervals out of the tonal default):
    python -m generation.tools.build_corpus_model --sources ingest:trance --out trance.json

    # or blend it into the general model by moving files up one level into
    # assets/raw/corpus/ (root), then rebuilding the default model.

You own the licensing of anything added here (see ../README.md).
