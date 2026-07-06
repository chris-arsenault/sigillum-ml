# Arabian / maqam corpus (separate model)

Drop MIDI / MusicXML maqam melodies here, then build a **separate** model so its
microtonal/modal intervals never dilute the tonal default:

    python -m generation.tools.build_corpus_model --sources ingest:arabian --out arabian.json

Empty by design until you add material (you own its licensing — see ../README.md).
