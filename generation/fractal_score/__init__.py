"""Fractalized whole-score generation: a learnable coarse-to-fine refinement ladder.

A score attribute is represented on a maskable multi-resolution grid. One shared
learned operator engraves a coarse skeleton into progressively finer detail, and
applying it recursively is the fractalization step. The first instantiated level
is harmony: a coarse pillar grid is refined into a full per-bar chord
progression, held out by composer against copy/unigram/bigram baselines.
"""
from generation.fractal_score.baselines import Bigram, BigramLM, CopyNearestPillar, Unigram
from generation.fractal_score.dataset import (
    DatasetError,
    Window,
    extract_windows,
    movement_holdout,
    refinement_arrays,
    split_windows,
)
from generation.fractal_score.harmony import (
    NONE_TOKEN,
    HarmonyCorpusError,
    HarmonyProgression,
    build_progressions,
)
from generation.fractal_score.ladder import (
    LadderError,
    RefinementSchedule,
    refinement_positions,
    revealed_positions,
)
from generation.fractal_score.model import RefinementConfig, RefinementOperator
from generation.fractal_score.refine import (
    GenerativeMetric,
    RealismMetric,
    RecursiveMetric,
    StepMetric,
    evaluate_generative,
    evaluate_realism,
    evaluate_recursive,
    evaluate_steps,
    recursive_refine,
)
from generation.fractal_score.train import TrainConfig, train_operator
from generation.fractal_score.vocab import HarmonyVocab, VocabError

__all__ = [
    "Bigram",
    "BigramLM",
    "CopyNearestPillar",
    "DatasetError",
    "GenerativeMetric",
    "HarmonyCorpusError",
    "HarmonyProgression",
    "HarmonyVocab",
    "LadderError",
    "NONE_TOKEN",
    "RecursiveMetric",
    "RealismMetric",
    "RefinementConfig",
    "RefinementOperator",
    "RefinementSchedule",
    "StepMetric",
    "TrainConfig",
    "Unigram",
    "VocabError",
    "Window",
    "build_progressions",
    "evaluate_generative",
    "evaluate_realism",
    "evaluate_recursive",
    "evaluate_steps",
    "extract_windows",
    "movement_holdout",
    "recursive_refine",
    "refinement_arrays",
    "refinement_positions",
    "revealed_positions",
    "split_windows",
    "train_operator",
]
