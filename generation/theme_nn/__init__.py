"""Neural theme model (in design). The first committed piece is the note ``representation``;
the model, training pipeline, and decode follow. The Markov generator in ``theme_gen`` is
the baseline this is meant to beat. See ``docs/architecture/18_feature_extraction.md`` for the
feature surface that feeds it."""
from __future__ import annotations

from generation.theme_nn.representation import Event, decode, encode

__all__ = ["Event", "encode", "decode"]
