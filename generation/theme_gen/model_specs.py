"""Durable, named model specifications.

A model spec pairs a name with a composable corpus spec (the same ``load_corpus`` tokens the
CLI takes, e.g. ``"exotic"``, ``"fiddle@150"``, ``"vgm"``) plus the model hyper-parameters
and an output filename. Specs are authored here (version-controlled, no JSON/YAML — the
repo's convention), so ``python -m generation.tools.build_corpus_model --spec <name>`` rebuilds a named
model reproducibly. The trained artifact lands in ``outputs/models/`` (rebuildable, ignored);
the *spec* is the durable record.

Add your own named models to ``MODEL_SPECS`` below.
"""
from __future__ import annotations

from dataclasses import dataclass

from generation.theme_gen.model import (
    DEFAULT_BEATS_PER_BAR,
    DEFAULT_MIN_COUNT,
    DEFAULT_ORDER,
)


@dataclass(frozen=True)
class ModelSpec:
    """A named model: which corpus to train on, and how."""

    sources: tuple[str, ...]                 # load_corpus tokens, e.g. ("exotic", "fiddle@150")
    description: str = ""
    out: str = ""                            # artifact filename; defaults to "<name>.json"
    order: int = DEFAULT_ORDER
    beats_per_bar: float = DEFAULT_BEATS_PER_BAR
    min_count: int = DEFAULT_MIN_COUNT
    sample_seed: int = 0

    def output_name(self, name: str) -> str:
        return self.out or f"{name}.json"


# The durable manifest of named models. Edit / extend freely.
MODEL_SPECS: dict[str, ModelSpec] = {
    "general_tonal": ModelSpec(
        ("bach", "essenFolksong", "fiddle"),
        description="Licensing-clean general-tonal default (chorales + folk + fiddle tunes).",
    ),
    "general_plus_vgm": ModelSpec(
        ("bach", "essenFolksong", "fiddle", "vgm"),
        description="General tonal blended with the extracted game-melody leads.",
    ),
    "exotic": ModelSpec(
        ("exotic", "fiddle@150"),
        description="Non-Western scales (Hijaz / Hungarian / double-harmonic / whole-tone) "
                    "grounded with a little tonal fiddle so it is not purely modal.",
    ),
    "arabian": ModelSpec(
        ("ingest:arabian",),
        description="Separate maqam model from user-dropped material (empty until populated).",
    ),
    "vgm_bach_classical": ModelSpec(
        ("vgm", "bach", "classical"),
        description="Game leads + Bach chorales + harvested classical melody.",
    ),
    "everything": ModelSpec(
        ("bach", "fiddle@600", "folk@1500", "vgm", "exotic", "classical"),
        description="Balanced blend of every category (folk/fiddle capped so none dominates).",
    ),
}


def get_spec(name: str) -> ModelSpec:
    if name not in MODEL_SPECS:
        raise KeyError(f"unknown model spec {name!r}; known: {', '.join(sorted(MODEL_SPECS))}")
    return MODEL_SPECS[name]
