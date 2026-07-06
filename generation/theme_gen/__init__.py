"""Kernel-constrained theme candidate generation package.

Split by concern from the original theme-generation module (behavior-preserving,
plan M0). Import public generator APIs from ``generation.theme_gen``.

See ``docs/project/THEME-GENERATOR-PLAN.md`` and
``docs/architecture/17_theme_generator.md``.
"""
from __future__ import annotations

from generation.theme_gen._common import EPS, MelodyItem, TICKS_PER_QUARTER
from generation.theme_gen.kerneldsl import (
    HarmonicPin,
    PitchRhythmPin,
    StructuralPin,
    ThemeFrame,
    ThemeKernel,
    frame,
    harm,
    kernel,
    phrase,
    pin,
)
from generation.theme_gen.corpus import default_theme_corpus
from generation.theme_gen.model import MarkovMelodyModel
from generation.theme_gen.features import (
    ConformanceWeights,
    FeatureSummary,
    TargetProfile,
    conformance_distance,
    measure_free_features,
    roll_target_profile,
)
from generation.theme_gen.engine import (
    CandidateAttempt,
    GenerationTrace,
    ThemeCandidate,
    generate_theme_batch,
    pins_satisfied,
)
from generation.theme_gen.report import render_candidate_report, render_run_log

__all__ = [
    "EPS",
    "MelodyItem",
    "TICKS_PER_QUARTER",
    "ThemeFrame",
    "PitchRhythmPin",
    "HarmonicPin",
    "StructuralPin",
    "ThemeKernel",
    "frame",
    "pin",
    "harm",
    "phrase",
    "kernel",
    "TargetProfile",
    "FeatureSummary",
    "ConformanceWeights",
    "ThemeCandidate",
    "CandidateAttempt",
    "GenerationTrace",
    "MarkovMelodyModel",
    "default_theme_corpus",
    "generate_theme_batch",
    "roll_target_profile",
    "measure_free_features",
    "conformance_distance",
    "pins_satisfied",
    "render_candidate_report",
    "render_run_log",
]
