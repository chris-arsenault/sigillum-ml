"""Paths owned by the Sigillum ML workspace.

These are application paths, not score-framework paths. Partitura remains the
authority for score compilation, analysis, and export.
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path:
    configured = os.environ.get("SIGILLUM_ML_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


ROOT = _project_root()
OUTPUTS = ROOT / "outputs"
SKETCH_OUTPUTS = OUTPUTS / "sketches"
MODEL_OUTPUTS = OUTPUTS / "models"
RAW_CORPUS = ROOT / "assets" / "raw" / "corpus"
WHOLE_SCORE_CORPUS = RAW_CORPUS / "whole_score"


def model_output(filename: str) -> Path:
    """Return a model artifact path, creating its ignored parent directory."""

    MODEL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    return MODEL_OUTPUTS / filename
