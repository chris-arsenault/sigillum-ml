"""Leakage-safe pairwise learning over Ruby-owned composition evidence.

Partitura owns scores, snapshots, review rendering, and human preferences. This
module treats those records as immutable observations. Its feature extractor is
deliberately schema-agnostic: it compares numeric and short categorical leaves
without assigning musical meaning to any field.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional

from generation.composition.evidence import (
    ORIGINAL_CANDIDATE_ID,
    CompositionDataset,
    PairwiseExample,
)
from generation.composition.observation_dataset import (
    _canonical_digest,
    _file_digest,
    _identifier,
    _list,
    _mapping,
    _positive_integer,
    _safe_relative,
    _string,
)

_CRITERIA = ("coherence", "identity", "seams", "orchestration", "reserve")
_SPLITS = ("train", "validation", "held_out_evaluation")
_SELECTION_METRIC = "mean_criterion_balanced_accuracy"
_IGNORED_FIELD_FRAGMENTS = (
    "artifact",
    "digest",
    "filename",
    "patch",
    "recorded_at",
)


class CriticLearningError(ValueError):
    """Raised when critic evidence, configuration, or artifacts are invalid."""


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CriticLearningError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CriticLearningError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise CriticLearningError(f"{label} must be at least {minimum}")
    if maximum is not None and result >= maximum:
        raise CriticLearningError(f"{label} must be less than {maximum}")
    return result


@dataclass(frozen=True)
class CriticCorpusSource:
    trajectory: PurePosixPath
    trajectory_digest: str
    reviews: PurePosixPath
    reviews_digest: str
    preferences: PurePosixPath
    preferences_digest: str

    @classmethod
    def from_dict(cls, value: object) -> CriticCorpusSource:
        data = _mapping(value, "critic corpus source")
        paths: dict[str, PurePosixPath] = {
            label: _safe_relative(data.get(label), f"critic source {label}")
            for label in ("trajectory", "reviews", "preferences")
        }
        if any(path.parts[0] != "outputs" for path in paths.values()):
            raise CriticLearningError("critic corpus sources must stay under outputs/")
        digests = {
            f"{label}_digest": _string(
                data.get(f"{label}_digest"),
                f"critic source {label}_digest",
            )
            for label in ("trajectory", "reviews", "preferences")
        }
        return cls(**paths, **digests)

    def to_dict(self) -> dict[str, str]:
        return {
            "trajectory": self.trajectory.as_posix(),
            "trajectory_digest": self.trajectory_digest,
            "reviews": self.reviews.as_posix(),
            "reviews_digest": self.reviews_digest,
            "preferences": self.preferences.as_posix(),
            "preferences_digest": self.preferences_digest,
        }


@dataclass(frozen=True)
class CriticCriterionSpec:
    id: str
    scales: tuple[str, ...]
    minimum_pairs: Mapping[str, int]
    minimum_runs: Mapping[str, int]

    @classmethod
    def from_dict(cls, value: object) -> CriticCriterionSpec:
        data = _mapping(value, "critic criterion")
        criterion_id = _identifier(data.get("id"), "critic criterion id")
        if criterion_id not in _CRITERIA:
            raise CriticLearningError(
                f"unsupported critic criterion {criterion_id!r}"
            )
        scales = tuple(
            _identifier(item, "critic scale")
            for item in _list(data.get("scales"), "critic scales")
        )
        if not scales or len(scales) != len(set(scales)):
            raise CriticLearningError("critic scales must be non-empty and unique")
        minimum_pairs = _mapping(data.get("minimum_pairs"), "minimum_pairs")
        minimum_runs = _mapping(data.get("minimum_runs"), "minimum_runs")
        for label, counts in (
            ("minimum_pairs", minimum_pairs),
            ("minimum_runs", minimum_runs),
        ):
            if set(counts) != set(_SPLITS):
                raise CriticLearningError(f"{label} must cover every critic split")
            for split, count in counts.items():
                _positive_integer(count, f"{label}.{split}")
        return cls(
            id=criterion_id,
            scales=scales,
            minimum_pairs=MappingProxyType(
                {split: int(minimum_pairs[split]) for split in _SPLITS}
            ),
            minimum_runs=MappingProxyType(
                {split: int(minimum_runs[split]) for split in _SPLITS}
            ),
        )


@dataclass(frozen=True)
class CriticLearningSpec:
    experiment_id: str
    corpus_index: PurePosixPath
    expected_corpus_index_digest: str | None
    output_root: PurePosixPath
    criteria: tuple[CriticCriterionSpec, ...]
    validation_salt: str
    validation_modulus: int
    validation_buckets: tuple[int, ...]
    seed: int
    representation_dim: int
    dropout: float
    steps: int
    batch_size_per_criterion: int
    learning_rate: float
    weight_decay: float
    validation_interval: int
    patience: int
    gradient_clip: float
    evaluation_batch_size: int
    selection_metric: str
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> CriticLearningSpec:
        spec_path = Path(path)
        try:
            value = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CriticLearningError(
                f"cannot read critic learning spec {spec_path}: {error}"
            ) from error
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: object) -> CriticLearningSpec:
        data = _mapping(value, "critic learning spec")
        if data.get("schema_version") != 1:
            raise CriticLearningError("critic learning spec schema_version must be 1")
        criteria = tuple(
            CriticCriterionSpec.from_dict(item)
            for item in _list(data.get("criteria"), "critic criteria")
        )
        if tuple(item.id for item in criteria) != _CRITERIA:
            raise CriticLearningError(
                "critic criteria must contain the five frozen criteria in order"
            )
        corpus_index = _safe_relative(data.get("corpus_index"), "corpus_index")
        output_root = _safe_relative(data.get("output_root"), "output_root")
        if corpus_index.parts[0] != "outputs" or output_root.parts[0] != "outputs":
            raise CriticLearningError(
                "critic corpus index and output root must stay under outputs/"
            )
        expected_digest = data.get("expected_corpus_index_digest")
        if expected_digest is not None:
            expected_digest = _string(
                expected_digest, "expected_corpus_index_digest"
            )
        validation = _mapping(data.get("validation"), "validation split")
        modulus = _positive_integer(validation.get("modulus"), "validation.modulus")
        buckets = tuple(
            _positive_integer(
                item,
                "validation bucket",
                allow_zero=True,
            )
            for item in _list(validation.get("buckets"), "validation buckets")
        )
        if not buckets or len(buckets) != len(set(buckets)):
            raise CriticLearningError(
                "validation buckets must be non-empty and unique"
            )
        if any(bucket >= modulus for bucket in buckets):
            raise CriticLearningError(
                "validation buckets must be less than the modulus"
            )
        model = _mapping(data.get("model"), "critic model")
        training = _mapping(data.get("training"), "critic training")
        selection_metric = _string(data.get("selection_metric"), "selection_metric")
        if selection_metric != _SELECTION_METRIC:
            raise CriticLearningError(
                f"selection_metric must be {_SELECTION_METRIC}"
            )
        dropout = _number(
            model.get("dropout"), "model.dropout", minimum=0, maximum=1
        )
        learning_rate = _number(
            training.get("learning_rate"), "training.learning_rate", minimum=0
        )
        gradient_clip = _number(
            training.get("gradient_clip"), "training.gradient_clip", minimum=0
        )
        if learning_rate == 0 or gradient_clip == 0:
            raise CriticLearningError(
                "learning_rate and gradient_clip must be positive"
            )
        steps = _positive_integer(training.get("steps"), "training.steps")
        validation_interval = _positive_integer(
            training.get("validation_interval"), "training.validation_interval"
        )
        if validation_interval > steps:
            raise CriticLearningError(
                "validation_interval cannot exceed training steps"
            )
        return cls(
            experiment_id=_identifier(data.get("experiment_id"), "experiment_id"),
            corpus_index=corpus_index,
            expected_corpus_index_digest=expected_digest,
            output_root=output_root,
            criteria=criteria,
            validation_salt=_string(validation.get("salt"), "validation.salt"),
            validation_modulus=modulus,
            validation_buckets=buckets,
            seed=_positive_integer(data.get("seed"), "seed", allow_zero=True),
            representation_dim=_positive_integer(
                model.get("representation_dim"), "model.representation_dim"
            ),
            dropout=dropout,
            steps=steps,
            batch_size_per_criterion=_positive_integer(
                training.get("batch_size_per_criterion"),
                "training.batch_size_per_criterion",
            ),
            learning_rate=learning_rate,
            weight_decay=_number(
                training.get("weight_decay"),
                "training.weight_decay",
                minimum=0,
            ),
            validation_interval=validation_interval,
            patience=_positive_integer(training.get("patience"), "training.patience"),
            gradient_clip=gradient_clip,
            evaluation_batch_size=_positive_integer(
                training.get("evaluation_batch_size"),
                "training.evaluation_batch_size",
            ),
            selection_metric=selection_metric,
            raw=json.loads(json.dumps(data)),
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.raw)

    @property
    def criteria_by_id(self) -> Mapping[str, CriticCriterionSpec]:
        return MappingProxyType({item.id: item for item in self.criteria})


@dataclass(frozen=True)
class CriticCorpusIndex:
    sources: tuple[CriticCorpusSource, ...]
    digest: str

    @classmethod
    def load(cls, path: str | Path) -> CriticCorpusIndex:
        index_path = Path(path)
        try:
            value = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CriticLearningError(
                f"cannot read critic corpus index {index_path}: {error}"
            ) from error
        data = _mapping(value, "critic corpus index")
        if data.get("schema_version") != 1:
            raise CriticLearningError("critic corpus index schema_version must be 1")
        sources = tuple(
            CriticCorpusSource.from_dict(item)
            for item in _list(data.get("sources"), "critic corpus sources")
        )
        if not sources or len(sources) != len(set(sources)):
            raise CriticLearningError(
                "critic corpus sources must be non-empty and unique"
            )
        claimed = _string(data.get("corpus_index_digest"), "corpus_index_digest")
        payload = dict(data)
        payload.pop("corpus_index_digest", None)
        actual = _canonical_digest(payload)
        if claimed != actual:
            raise CriticLearningError("critic corpus index digest mismatch")
        return cls(sources=sources, digest=claimed)


@dataclass(frozen=True)
class CriticPair:
    id: str
    criterion: str
    scale: str
    run_id: str
    split: str
    label: int
    features: Mapping[str, float]


@dataclass(frozen=True)
class PreparedCriticSplit:
    values: np.ndarray
    labels: np.ndarray
    pair_ids: tuple[str, ...]
    run_ids: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.pair_ids)


@dataclass(frozen=True)
class PreparedCriticCorpus:
    corpus_index_digest: str
    feature_vocabulary: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    splits: Mapping[str, Mapping[str, PreparedCriticSplit]]
    audit: Mapping[str, Any]


class OpaquePairFeatureExtractor:
    """Compare opaque snapshots without defining music in Python."""

    def extract(self, pair: PairwiseExample) -> Mapping[str, float]:
        left_id = pair.review.variants["A"]
        right_id = pair.review.variants["B"]
        left = self._candidate_features(pair, left_id)
        right = self._candidate_features(pair, right_id)
        names = sorted(set(left) | set(right))
        difference = {
            name: float(left.get(name, 0.0) - right.get(name, 0.0))
            for name in names
            if not math.isclose(
                left.get(name, 0.0),
                right.get(name, 0.0),
                abs_tol=1e-12,
            )
        }
        if not difference:
            raise CriticLearningError(
                f"preference {pair.preference.preference_id} has no observable "
                "snapshot difference"
            )
        return MappingProxyType(difference)

    def _candidate_features(
        self,
        pair: PairwiseExample,
        candidate_id: str,
    ) -> Mapping[str, float]:
        if candidate_id == ORIGINAL_CANDIDATE_ID:
            snapshot = pair.transition.before_snapshot
        else:
            assessment = next(
                (
                    item
                    for item in pair.transition.candidates
                    if _mapping(item.get("candidate"), "candidate").get("candidate_id")
                    == candidate_id
                ),
                None,
            )
            if assessment is None:
                raise CriticLearningError(
                    f"candidate {candidate_id} is absent from its transition"
                )
            snapshot = _mapping(
                assessment.get("candidate_snapshot"), "candidate_snapshot"
            )
        atoms: dict[str, list[float]] = defaultdict(list)
        self._collect(snapshot, (), atoms)
        return MappingProxyType(
            {
                name: sum(values) / len(values)
                for name, values in sorted(atoms.items())
            }
        )

    def _collect(
        self,
        value: object,
        path: tuple[str, ...],
        atoms: dict[str, list[float]],
    ) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                key_text = str(key)
                if self._ignored(key_text):
                    continue
                self._collect(value[key], (*path, key_text), atoms)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._collect(item, (*path, "[]"), atoms)
            return
        if not path:
            return
        name = ".".join(path)
        if isinstance(value, bool):
            atoms[name].append(float(value))
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            atoms[name].append(float(value))
        elif isinstance(value, str) and 0 < len(value) <= 64:
            atoms[f"{name}=={value}"].append(1.0)

    @staticmethod
    def _ignored(key: str) -> bool:
        normalized = key.lower()
        return (
            normalized == "id"
            or normalized.endswith("_id")
            or normalized.startswith("source")
            or any(
                fragment in normalized
                for fragment in _IGNORED_FIELD_FRAGMENTS
            )
        )


class CriticCorpusPreparer:
    def __init__(
        self,
        spec: CriticLearningSpec,
        pairs: Sequence[PairwiseExample],
        *,
        corpus_index_digest: str,
        extractor: OpaquePairFeatureExtractor | None = None,
    ):
        self.spec = spec
        self.pairs = tuple(pairs)
        self.corpus_index_digest = corpus_index_digest
        self.extractor = extractor or OpaquePairFeatureExtractor()

    def prepare(self, *, require_ready: bool = True) -> PreparedCriticCorpus:
        critic_pairs, feature_failures = self._pairs()
        audit = self._audit(critic_pairs, feature_failures)
        if require_ready and not audit["ready"]:
            raise CriticLearningError(
                "critic corpus does not satisfy its frozen readiness gates"
            )
        if not critic_pairs:
            return PreparedCriticCorpus(
                corpus_index_digest=self.corpus_index_digest,
                feature_vocabulary=(),
                feature_mean=_immutable_array(np.asarray([], dtype=np.float32)),
                feature_scale=_immutable_array(np.asarray([], dtype=np.float32)),
                splits=MappingProxyType({}),
                audit=MappingProxyType(audit),
            )
        train_pairs = [item for item in critic_pairs if item.split == "train"]
        vocabulary = tuple(
            sorted({name for item in train_pairs for name in item.features})
        )
        if not vocabulary:
            raise CriticLearningError("critic training pairs have no usable features")
        feature_index = {name: index for index, name in enumerate(vocabulary)}
        train_matrix = self._matrix(train_pairs, feature_index)
        # Pair orientation is augmented during training. The implied
        # distribution contains x and -x, so its train-only mean is exactly
        # zero and its scale is the root mean square.
        mean = np.zeros(len(vocabulary), dtype=np.float32)
        scale = np.sqrt(np.square(train_matrix).mean(axis=0))
        scale[scale == 0.0] = 1.0
        by_criterion: dict[str, Mapping[str, PreparedCriticSplit]] = {}
        for criterion in _CRITERIA:
            split_map = {}
            for split in _SPLITS:
                selected = [
                    item
                    for item in critic_pairs
                    if item.criterion == criterion and item.split == split
                ]
                matrix = self._matrix(selected, feature_index)
                matrix = (matrix - mean[None, :]) / scale[None, :]
                split_map[split] = PreparedCriticSplit(
                    values=_immutable_array(matrix.astype(np.float32)),
                    labels=_immutable_array(
                        np.asarray([item.label for item in selected], dtype=np.int64)
                    ),
                    pair_ids=tuple(item.id for item in selected),
                    run_ids=tuple(item.run_id for item in selected),
                )
            by_criterion[criterion] = MappingProxyType(split_map)
        return PreparedCriticCorpus(
            corpus_index_digest=self.corpus_index_digest,
            feature_vocabulary=vocabulary,
            feature_mean=_immutable_array(mean.astype(np.float32)),
            feature_scale=_immutable_array(scale.astype(np.float32)),
            splits=MappingProxyType(by_criterion),
            audit=MappingProxyType(audit),
        )

    def _pairs(self) -> tuple[list[CriticPair], list[dict[str, str]]]:
        prepared = []
        failures = []
        for pair in self.pairs:
            split = self._split(pair)
            try:
                features = self.extractor.extract(pair)
            except CriticLearningError as error:
                failures.append(
                    {
                        "preference_id": pair.preference.preference_id,
                        "error": str(error),
                    }
                )
                continue
            prepared.append(
                CriticPair(
                    id=pair.preference.preference_id,
                    criterion=pair.preference.criterion,
                    scale=pair.review.scale,
                    run_id=pair.transition.run_id,
                    split=split,
                    label=int(pair.preference.outcome == "a"),
                    features=features,
                )
            )
        return prepared, failures

    def _split(self, pair: PairwiseExample) -> str:
        if pair.preference.purpose == "held_out_evaluation":
            return "held_out_evaluation"
        digest = hashlib.sha256(
            f"{self.spec.validation_salt}\0{pair.transition.run_id}".encode()
        ).digest()
        bucket = int.from_bytes(digest[:8], "big") % self.spec.validation_modulus
        return "validation" if bucket in self.spec.validation_buckets else "train"

    def _audit(
        self,
        pairs: Sequence[CriticPair],
        feature_failures: Sequence[Mapping[str, str]],
    ) -> dict[str, Any]:
        run_splits: dict[str, set[str]] = defaultdict(set)
        pair_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
        raw_by_preference = {
            pair.preference.preference_id: pair for pair in self.pairs
        }
        for item in pairs:
            run_splits[item.run_id].add(item.split)
            source = raw_by_preference[item.id]
            candidate_ids = tuple(sorted(source.review.variants.values()))
            pair_splits[(source.transition.transition_id, *candidate_ids)].add(
                item.split
            )
        run_leakage = {
            run_id: sorted(splits)
            for run_id, splits in sorted(run_splits.items())
            if "held_out_evaluation" in splits and len(splits) > 1
        }
        comparison_leakage = [
            {
                "transition_id": key[0],
                "candidate_ids": list(key[1:]),
                "splits": sorted(splits),
            }
            for key, splits in sorted(pair_splits.items())
            if len(splits) > 1
        ]
        criteria = []
        all_criteria_ready = True
        by_id = self.spec.criteria_by_id
        for criterion in _CRITERIA:
            criterion_pairs = [item for item in pairs if item.criterion == criterion]
            counts = {
                split: sum(item.split == split for item in criterion_pairs)
                for split in _SPLITS
            }
            run_counts = {
                split: len(
                    {
                        item.run_id
                        for item in criterion_pairs
                        if item.split == split
                    }
                )
                for split in _SPLITS
            }
            labels = {
                split: sorted(
                    {
                        item.label
                        for item in criterion_pairs
                        if item.split == split
                    }
                )
                for split in _SPLITS
            }
            invalid_scales = sorted(
                {
                    item.scale
                    for item in criterion_pairs
                    if item.scale not in by_id[criterion].scales
                }
            )
            ready = (
                not invalid_scales
                and all(
                    counts[split] >= by_id[criterion].minimum_pairs[split]
                    and run_counts[split] >= by_id[criterion].minimum_runs[split]
                    and labels[split] == [0, 1]
                    for split in _SPLITS
                )
            )
            all_criteria_ready &= ready
            criteria.append(
                {
                    "criterion": criterion,
                    "ready": ready,
                    "pair_counts": counts,
                    "run_counts": run_counts,
                    "blind_outcomes": labels,
                    "invalid_scales": invalid_scales,
                }
            )
        return {
            "schema_version": 1,
            "kind": "critic_corpus_readiness",
            "experiment_id": self.spec.experiment_id,
            "spec_digest": self.spec.digest,
            "corpus_index_digest": self.corpus_index_digest,
            "ready": (
                all_criteria_ready
                and not run_leakage
                and not comparison_leakage
                and not feature_failures
            ),
            "resolved_preference_count": len(self.pairs),
            "usable_pair_count": len(pairs),
            "feature_failures": list(feature_failures),
            "run_leakage": run_leakage,
            "comparison_leakage": comparison_leakage,
            "criteria": criteria,
        }

    @staticmethod
    def _matrix(
        pairs: Sequence[CriticPair],
        feature_index: Mapping[str, int],
    ) -> np.ndarray:
        matrix = np.zeros((len(pairs), len(feature_index)), dtype=np.float32)
        for row, pair in enumerate(pairs):
            for name, value in pair.features.items():
                column = feature_index.get(name)
                if column is not None:
                    matrix[row, column] = value
        return matrix


class PairwiseCritic(nn.Module):
    """Shared learned representation with criterion-specific preference heads."""

    def __init__(
        self,
        *,
        feature_count: int,
        representation_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(feature_count),
            nn.Linear(feature_count, representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
        )
        self.heads = nn.ModuleDict(
            {criterion: nn.Linear(representation_dim, 1) for criterion in _CRITERIA}
        )

    def forward(
        self,
        *,
        criterion: str,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encoder(values)
        return self.heads[criterion](representation).squeeze(-1), representation


class PairwiseCriticTrainer:
    def __init__(
        self,
        spec: CriticLearningSpec,
        corpus: PreparedCriticCorpus,
    ):
        self.spec = spec
        self.corpus = corpus

    def train(self, *, output_root: str | Path) -> Mapping[str, Any]:
        if not self.corpus.audit.get("ready"):
            raise CriticLearningError("critic corpus is not ready for training")
        if (
            self.spec.expected_corpus_index_digest is None
            or self.spec.expected_corpus_index_digest
            != self.corpus.corpus_index_digest
        ):
            raise CriticLearningError(
                "critic corpus index is not pinned by the experiment"
            )
        _set_determinism(self.spec.seed)
        model = PairwiseCritic(
            feature_count=len(self.corpus.feature_vocabulary),
            representation_dim=self.spec.representation_dim,
            dropout=self.spec.dropout,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.spec.learning_rate,
            weight_decay=self.spec.weight_decay,
        )
        generator = np.random.default_rng(self.spec.seed)
        best_state: dict[str, torch.Tensor] | None = None
        best_validation: Mapping[str, Any] | None = None
        best_score = -math.inf
        best_step = 0
        without_improvement = 0
        history = []
        completed_steps = 0
        for step in range(1, self.spec.steps + 1):
            loss = self._training_step(model, optimizer, generator)
            completed_steps = step
            if (
                step % self.spec.validation_interval != 0
                and step != self.spec.steps
            ):
                continue
            validation = self._evaluate(model, "validation")
            score = _mean_balanced_accuracy(validation)
            history.append(
                {
                    "step": step,
                    "training_loss": round(loss, 8),
                    "selection_score": round(score, 8),
                }
            )
            if score > best_score:
                best_score = score
                best_step = step
                best_validation = validation
                best_state = _clone_state(model.state_dict())
                without_improvement = 0
            else:
                without_improvement += 1
            if without_improvement >= self.spec.patience:
                break
        if best_state is None or best_validation is None:
            raise CriticLearningError(
                "critic training completed without a validation checkpoint"
            )
        model.load_state_dict(best_state)
        held_out = self._evaluate(model, "held_out_evaluation")
        model_digest = _model_digest(best_state)
        output = Path(output_root)
        checkpoint_path = output / "checkpoint.pt"
        checkpoint = {
            "schema_version": 1,
            "experiment_id": self.spec.experiment_id,
            "spec_digest": self.spec.digest,
            "corpus_index_digest": self.corpus.corpus_index_digest,
            "model_digest": model_digest,
            "best_step": best_step,
            "feature_vocabulary": self.corpus.feature_vocabulary,
            "feature_mean": torch.tensor(np.asarray(self.corpus.feature_mean)),
            "feature_scale": torch.tensor(np.asarray(self.corpus.feature_scale)),
            "model": {
                "representation_dim": self.spec.representation_dim,
                "dropout": self.spec.dropout,
            },
            "state_dict": best_state,
        }
        _write_torch(checkpoint_path, checkpoint)
        report: dict[str, Any] = {
            "schema_version": 1,
            "kind": "pairwise_critic_training_report",
            "experiment_id": self.spec.experiment_id,
            "spec_digest": self.spec.digest,
            "corpus_index_digest": self.corpus.corpus_index_digest,
            "model_digest": model_digest,
            "checkpoint_digest": _file_digest(checkpoint_path),
            "selection_metric": self.spec.selection_metric,
            "feature_vocabulary_size": len(self.corpus.feature_vocabulary),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "best_step": best_step,
            "completed_steps": completed_steps,
            "validation_history": history,
            "best_validation_score": round(
                _mean_balanced_accuracy(best_validation), 8
            ),
            "held_out_score": round(_mean_balanced_accuracy(held_out), 8),
            "criteria": [
                {
                    "criterion": criterion,
                    "validation": best_validation[criterion],
                    "held_out_evaluation": held_out[criterion],
                }
                for criterion in _CRITERIA
            ],
        }
        report["report_digest"] = _canonical_digest(report)
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "report.json", report)
        return report

    def _training_step(
        self,
        model: PairwiseCritic,
        optimizer: torch.optim.Optimizer,
        generator: np.random.Generator,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for criterion in _CRITERIA:
            split = self.corpus.splits[criterion]["train"]
            batches = []
            labels = []
            half = max(1, self.spec.batch_size_per_criterion // 2)
            indices = generator.integers(0, split.count, size=half)
            for index in indices:
                value = np.asarray(split.values[index])
                label = int(split.labels[index])
                batches.extend((value, -value))
                labels.extend((label, 1 - label))
            logits, _ = model(
                criterion=criterion,
                values=torch.tensor(np.asarray(batches), dtype=torch.float32),
            )
            target = torch.tensor(labels, dtype=torch.float32)
            losses.append(
                torch_functional.binary_cross_entropy_with_logits(logits, target)
            )
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.spec.gradient_clip)
        optimizer.step()
        return float(loss.detach())

    def _evaluate(
        self,
        model: PairwiseCritic,
        split_name: str,
    ) -> dict[str, dict[str, Any]]:
        model.eval()
        results = {}
        with torch.inference_mode():
            for criterion in _CRITERIA:
                split = self.corpus.splits[criterion][split_name]
                probabilities = []
                for start in range(0, split.count, self.spec.evaluation_batch_size):
                    end = min(start + self.spec.evaluation_batch_size, split.count)
                    logits, _ = model(
                        criterion=criterion,
                        values=torch.tensor(
                            np.asarray(split.values[start:end]),
                            dtype=torch.float32,
                        ),
                    )
                    probabilities.extend(torch.sigmoid(logits).tolist())
                results[criterion] = _preference_metrics(
                    split.labels.tolist(), probabilities
                )
        return results


def load_critic_pairs(
    *,
    project_root: str | Path,
    index: CriticCorpusIndex,
) -> tuple[PairwiseExample, ...]:
    root = Path(project_root).resolve()
    pairs = []
    for source in index.sources:
        paths = {
            label: root.joinpath(*getattr(source, label).parts)
            for label in ("trajectory", "reviews", "preferences")
        }
        for label, path in paths.items():
            expected = getattr(source, f"{label}_digest")
            if _file_digest(path) != expected:
                raise CriticLearningError(
                    f"critic source digest mismatch for {source.to_dict()[label]}"
                )
        dataset = CompositionDataset.from_jsonl(
            paths["trajectory"],
            paths["reviews"],
            paths["preferences"],
        )
        pairs.extend(dataset.training_pairs())
        pairs.extend(dataset.held_out_pairs())
    return tuple(pairs)


def build_critic_corpus_index(
    *,
    project_root: str | Path,
    sources: Sequence[Sequence[str]],
    output_path: str | Path,
) -> Mapping[str, Any]:
    root = Path(project_root).resolve()
    records = []
    for source_number, source in enumerate(sources, start=1):
        if len(source) != 3:
            raise CriticLearningError(
                f"critic source {source_number} must provide three paths"
            )
        data: dict[str, str] = {}
        for label, raw_path in zip(
            ("trajectory", "reviews", "preferences"),
            source,
        ):
            relative = _safe_relative(raw_path, f"critic source {label}")
            if relative.parts[0] != "outputs":
                raise CriticLearningError(
                    "critic corpus sources must stay under outputs/"
                )
            path = root.joinpath(*relative.parts)
            if not path.is_file():
                raise CriticLearningError(
                    f"critic source is absent: {relative.as_posix()}"
                )
            data[label] = relative.as_posix()
            data[f"{label}_digest"] = _file_digest(path)
        records.append(data)
    if not records or len({json.dumps(item, sort_keys=True) for item in records}) != len(
        records
    ):
        raise CriticLearningError(
            "critic corpus index sources must be non-empty and unique"
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "sources": records,
    }
    payload["corpus_index_digest"] = _canonical_digest(payload)
    _write_json(Path(output_path), payload)
    return payload


def critic_readiness_without_index(
    spec: CriticLearningSpec,
    *,
    reason: str,
) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "kind": "critic_corpus_readiness",
        "experiment_id": spec.experiment_id,
        "spec_digest": spec.digest,
        "corpus_index_digest": None,
        "ready": False,
        "reason": reason,
        "resolved_preference_count": 0,
        "usable_pair_count": 0,
        "feature_failures": [],
        "run_leakage": {},
        "comparison_leakage": [],
        "criteria": [
            {
                "criterion": criterion.id,
                "ready": False,
                "pair_counts": {split: 0 for split in _SPLITS},
                "run_counts": {split: 0 for split in _SPLITS},
                "blind_outcomes": {split: [] for split in _SPLITS},
                "invalid_scales": [],
            }
            for criterion in spec.criteria
        ],
    }


def verify_critic_artifacts(
    *,
    spec: CriticLearningSpec,
    checkpoint_path: str | Path,
    report_path: str | Path,
) -> Mapping[str, Any]:
    checkpoint_file = Path(checkpoint_path)
    report_file = Path(report_path)
    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=True,
        )
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        raise CriticLearningError(f"cannot read critic artifacts: {error}") from error
    checkpoint_data = _mapping(checkpoint, "critic checkpoint")
    report_data = _mapping(report, "critic report")
    claimed_report = _string(report_data.get("report_digest"), "report_digest")
    report_payload = dict(report_data)
    report_payload.pop("report_digest", None)
    report_payload.pop("generated_at", None)
    if _canonical_digest(report_payload) != claimed_report:
        raise CriticLearningError("critic report digest mismatch")
    state = _mapping(checkpoint_data.get("state_dict"), "critic state_dict")
    actual_model_digest = _model_digest(state)
    expected = {
        "spec_digest": spec.digest,
        "corpus_index_digest": spec.expected_corpus_index_digest,
        "model_digest": actual_model_digest,
    }
    for key, value in expected.items():
        if checkpoint_data.get(key) != value or report_data.get(key) != value:
            raise CriticLearningError(f"critic artifact lineage mismatch for {key}")
    if report_data.get("checkpoint_digest") != _file_digest(checkpoint_file):
        raise CriticLearningError("critic checkpoint file digest mismatch")
    return {
        "status": "ok",
        "experiment_id": spec.experiment_id,
        "model_digest": actual_model_digest,
        "report_digest": claimed_report,
        "best_step": checkpoint_data.get("best_step"),
    }


def _preference_metrics(
    expected: Sequence[int],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    if len(expected) != len(probabilities) or not expected:
        raise CriticLearningError("critic predictions do not align")
    predicted = [int(value >= 0.5) for value in probabilities]
    recalls = []
    for label in (0, 1):
        indices = [index for index, value in enumerate(expected) if value == label]
        if not indices:
            raise CriticLearningError(
                "critic evaluation split lacks both blinded outcomes"
            )
        recalls.append(
            sum(predicted[index] == label for index in indices) / len(indices)
        )
    return {
        "example_count": len(expected),
        "accuracy": round(
            sum(left == right for left, right in zip(expected, predicted))
            / len(expected),
            6,
        ),
        "balanced_accuracy": round(sum(recalls) / 2, 6),
        "brier": round(
            sum(
                (float(label) - probability) ** 2
                for label, probability in zip(expected, probabilities)
            )
            / len(expected),
            6,
        ),
    }


def _mean_balanced_accuracy(metrics: Mapping[str, Any]) -> float:
    return sum(
        float(value["balanced_accuracy"]) for value in metrics.values()
    ) / len(metrics)


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _immutable_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _clone_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in state.items()
    }


def _model_digest(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise CriticLearningError(
                f"critic state value is not a tensor: {name}"
            )
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _write_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        torch.save(value, temporary)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
