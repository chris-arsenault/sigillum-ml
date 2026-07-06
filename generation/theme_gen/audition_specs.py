"""Durable, named audition specifications.

An audition pairs a kernel with a set of trained models to render it through, so you can
produce a side-by-side survey of one theme across several corpora with a single command
(no ad-hoc CLI). Authored here (version-controlled), run with::

    python -m generation.tools.gen_themes --audition pulse_survey
    python -m generation.tools.gen_themes --list-auditions

Output is one combined audition score (each block labelled ``<model> CAND <n>``) plus a
report, written under ``outputs/sketches/auditions/``.
"""
from __future__ import annotations

from dataclasses import dataclass

# The full set of built model specs, in a sensible listening order.
ALL_MODELS = (
    "general_tonal",
    "general_plus_vgm",
    "vgm_bach_classical",
    "exotic",
    "arabian",
    "everything",
)


@dataclass(frozen=True)
class AuditionSpec:
    """Render ``kernel`` through each model in ``models``, ``batch`` candidates apiece."""

    kernel: str                                  # dotted module path or .py file exposing KERNEL
    models: tuple[str, ...] = ALL_MODELS         # model spec names (or .json paths)
    pool: int = 48
    batch: int = 3
    seed: int = 0
    tempo: int = 120
    description: str = ""


AUDITIONS: dict[str, AuditionSpec] = {
    "pulse_survey": AuditionSpec(
        kernel="experiments.pulse_kernel",
        models=("general_tonal", "vgm_bach_classical", "exotic", "arabian", "everything"),
        pool=48,
        batch=3,
        seed=7,
        tempo=140,
        description="The Pulse (Mvt III) battle kernel rendered across the trained models.",
    ),
    "m1_survey": AuditionSpec(
        kernel="experiments.m1_kernel",
        models=("general_tonal", "vgm_bach_classical", "everything"),
        pool=48,
        batch=3,
        seed=7,
        tempo=72,
        description="The Invocation (Mvt I) F-major head kernel across the tonal models.",
    ),
}


def get_audition(name: str) -> AuditionSpec:
    if name not in AUDITIONS:
        raise KeyError(f"unknown audition {name!r}; known: {', '.join(sorted(AUDITIONS))}")
    return AUDITIONS[name]
