"""Self-supervised structural-context learning over Partitura observations.

Ruby Partitura owns score meaning and emits the canonical observation. This
module performs only ML-side tensorization: it turns those explicit note, rest,
part, measure, pitch, and timing facts into model inputs and learns whether a
candidate span is the authentic continuation of an anchor span.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as torch_functional

from generation.composition.observation_dataset import (
    ObservationDataset,
    ScoreObservation,
    _canonical_digest,
)
from generation.composition.protocol import (
    LearnedCriticResult,
    SelectionRequest,
)


class StructuralContextError(ValueError):
    """Raised when the experiment or its Partitura observations are invalid."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StructuralContextError(f"{label} must be an object")
    return value


def _positive_integer(
    value: object, label: str, *, allow_zero: bool = False
) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise StructuralContextError(f"{label} must be a {qualifier} integer")
    return value


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise StructuralContextError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise StructuralContextError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise StructuralContextError(f"{label} must be at least {minimum}")
    if maximum is not None and result >= maximum:
        raise StructuralContextError(f"{label} must be less than {maximum}")
    return result


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StructuralContextError(f"{label} must be a non-empty string")
    return value


def _safe_relative(value: object, label: str) -> PurePosixPath:
    path = PurePosixPath(_nonempty_string(value, label))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise StructuralContextError(f"{label} must be a safe relative path")
    return path


def _rational(value: object, label: str) -> float:
    try:
        result = float(Fraction(str(value)))
    except (ValueError, ZeroDivisionError) as error:
        raise StructuralContextError(f"{label} must be rational") from error
    if not math.isfinite(result):
        raise StructuralContextError(f"{label} must be finite")
    return result


def _freeze(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class StructuralContextSpec:
    experiment_id: str
    observation_manifest: PurePosixPath
    expected_observation_manifest_digest: str
    output_root: PurePosixPath
    seed: int
    span_measures: int
    stride_measures: int
    minimum_negative_distance_measures: int
    maximum_examples_per_score: int
    negative_sampling: str
    maximum_parts: int
    onset_bins: int
    register_bins: int
    duration_ratio_boundaries: tuple[float, ...]
    measure_dimension: int
    span_dimension: int
    score_hidden_dimension: int
    dropout: float
    baseline_residual: bool
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_interval: int
    patience: int
    gradient_clip: float
    evaluation_batch_size: int
    evaluate_test: bool
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> StructuralContextSpec:
        spec_path = Path(path)
        try:
            value = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StructuralContextError(
                f"cannot read structural-context spec {spec_path}: {error}"
            ) from error
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: object) -> StructuralContextSpec:
        data = _mapping(value, "structural-context spec")
        if data.get("schema_version") != 1:
            raise StructuralContextError(
                "structural-context spec schema_version must be 1"
            )
        task = _mapping(data.get("task"), "task")
        vectorizer = _mapping(data.get("vectorizer"), "vectorizer")
        model = _mapping(data.get("model"), "model")
        training = _mapping(data.get("training"), "training")
        boundaries_value = vectorizer.get("duration_ratio_boundaries")
        if not isinstance(boundaries_value, list):
            raise StructuralContextError(
                "vectorizer.duration_ratio_boundaries must be a list"
            )
        boundaries = tuple(
            _finite_number(
                item,
                "vectorizer.duration_ratio_boundaries item",
                minimum=0,
            )
            for item in boundaries_value
        )
        if not boundaries or any(
            left >= right for left, right in zip(boundaries, boundaries[1:])
        ):
            raise StructuralContextError(
                "duration ratio boundaries must be strictly increasing"
            )
        selection_metric = _nonempty_string(
            data.get("selection_metric"), "selection_metric"
        )
        if selection_metric != "validation_score_macro_accuracy":
            raise StructuralContextError(
                "selection_metric must be validation_score_macro_accuracy"
            )
        baseline = _nonempty_string(data.get("baseline"), "baseline")
        if baseline != "boundary_profile_distance":
            raise StructuralContextError(
                "baseline must be boundary_profile_distance"
            )
        negative_sampling = _nonempty_string(
            task.get("negative_sampling", "random_same_score"),
            "task.negative_sampling",
        )
        if negative_sampling not in {
            "random_same_score",
            "boundary_matched_same_score",
        }:
            raise StructuralContextError(
                "task.negative_sampling must be random_same_score or "
                "boundary_matched_same_score"
            )
        output_root = _safe_relative(data.get("output_root"), "output_root")
        if output_root.parts[0] != "outputs":
            raise StructuralContextError("output_root must stay under outputs/")
        evaluate_test = training.get("evaluate_test")
        if not isinstance(evaluate_test, bool):
            raise StructuralContextError("training.evaluate_test must be boolean")
        baseline_residual = model.get("baseline_residual", False)
        if not isinstance(baseline_residual, bool):
            raise StructuralContextError(
                "model.baseline_residual must be boolean"
            )
        steps = _positive_integer(training.get("steps"), "training.steps")
        validation_interval = _positive_integer(
            training.get("validation_interval"),
            "training.validation_interval",
        )
        if validation_interval > steps:
            raise StructuralContextError(
                "training.validation_interval cannot exceed training.steps"
            )
        return cls(
            experiment_id=_nonempty_string(
                data.get("experiment_id"), "experiment_id"
            ),
            observation_manifest=_safe_relative(
                data.get("observation_manifest"), "observation_manifest"
            ),
            expected_observation_manifest_digest=_nonempty_string(
                data.get("expected_observation_manifest_digest"),
                "expected_observation_manifest_digest",
            ),
            output_root=output_root,
            seed=_positive_integer(data.get("seed"), "seed", allow_zero=True),
            span_measures=_positive_integer(
                task.get("span_measures"), "task.span_measures"
            ),
            stride_measures=_positive_integer(
                task.get("stride_measures"), "task.stride_measures"
            ),
            minimum_negative_distance_measures=_positive_integer(
                task.get("minimum_negative_distance_measures"),
                "task.minimum_negative_distance_measures",
            ),
            maximum_examples_per_score=_positive_integer(
                task.get("maximum_examples_per_score"),
                "task.maximum_examples_per_score",
            ),
            negative_sampling=negative_sampling,
            maximum_parts=_positive_integer(
                vectorizer.get("maximum_parts"), "vectorizer.maximum_parts"
            ),
            onset_bins=_positive_integer(
                vectorizer.get("onset_bins"), "vectorizer.onset_bins"
            ),
            register_bins=_positive_integer(
                vectorizer.get("register_bins"), "vectorizer.register_bins"
            ),
            duration_ratio_boundaries=boundaries,
            measure_dimension=_positive_integer(
                model.get("measure_dimension"), "model.measure_dimension"
            ),
            span_dimension=_positive_integer(
                model.get("span_dimension"), "model.span_dimension"
            ),
            score_hidden_dimension=_positive_integer(
                model.get("score_hidden_dimension"),
                "model.score_hidden_dimension",
            ),
            dropout=_finite_number(
                model.get("dropout"), "model.dropout", minimum=0, maximum=1
            ),
            baseline_residual=baseline_residual,
            steps=steps,
            batch_size=_positive_integer(
                training.get("batch_size"), "training.batch_size"
            ),
            learning_rate=_finite_number(
                training.get("learning_rate"),
                "training.learning_rate",
                minimum=0,
            ),
            weight_decay=_finite_number(
                training.get("weight_decay"),
                "training.weight_decay",
                minimum=0,
            ),
            validation_interval=validation_interval,
            patience=_positive_integer(
                training.get("patience"), "training.patience"
            ),
            gradient_clip=_finite_number(
                training.get("gradient_clip"),
                "training.gradient_clip",
                minimum=0,
            ),
            evaluation_batch_size=_positive_integer(
                training.get("evaluation_batch_size"),
                "training.evaluation_batch_size",
            ),
            evaluate_test=evaluate_test,
            raw=json.loads(json.dumps(data)),
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.raw)


@dataclass(frozen=True)
class PreparedContextSplit:
    anchors: np.ndarray
    authentic: np.ndarray
    nonadjacent: np.ndarray
    score_ids: tuple[str, ...]
    anchor_starts: np.ndarray
    authentic_starts: np.ndarray
    nonadjacent_starts: np.ndarray

    @property
    def count(self) -> int:
        return len(self.score_ids)


@dataclass(frozen=True)
class PreparedStructuralContextDataset:
    observation_manifest_digest: str
    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    splits: Mapping[str, PreparedContextSplit]
    score_counts: Mapping[str, int]
    omitted_scores: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class StructuralSeamSignal:
    observation_digest: str
    adjacency_count: int
    learned_mean: float
    learned_tenth_percentile: float
    learned_minimum: float
    boundary_mean: float
    residual_mean: float
    worst_successor_start_position: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_digest": self.observation_digest,
            "adjacency_count": self.adjacency_count,
            "learned_mean": self.learned_mean,
            "learned_tenth_percentile": self.learned_tenth_percentile,
            "learned_minimum": self.learned_minimum,
            "boundary_mean": self.boundary_mean,
            "residual_mean": self.residual_mean,
            "worst_successor_start_position": (
                self.worst_successor_start_position
            ),
        }


class PartituraMeasureVectorizer:
    """Tensorize explicit Partitura events without parsing score source."""

    def __init__(self, spec: StructuralContextSpec):
        self.spec = spec
        duration_bins = len(spec.duration_ratio_boundaries) + 1
        scalar_names = (
            "log_note_count",
            "log_unpitched_count",
            "log_rest_count",
            "active_part_fraction",
            "mean_midi",
            "midi_standard_deviation",
            "midi_range",
            "mean_duration_ratio",
            "rest_event_fraction",
            "log_measure_duration",
        )
        self.feature_names = (
            tuple(f"pitch_class_{index}" for index in range(12))
            + tuple(
                f"register_bin_{index}" for index in range(spec.register_bins)
            )
            + tuple(f"onset_bin_{index}" for index in range(spec.onset_bins))
            + tuple(f"duration_bin_{index}" for index in range(duration_bins))
            + tuple(
                f"part_{index}_active" for index in range(spec.maximum_parts)
            )
            + tuple(
                f"part_{index}_sounding_event_share"
                for index in range(spec.maximum_parts)
            )
            + scalar_names
        )

    def vectorize(self, observation: ScoreObservation) -> np.ndarray:
        score = _mapping(observation.data.get("score"), "score")
        measures = score.get("measures")
        parts = score.get("parts")
        events = score.get("timed_events")
        if not isinstance(measures, list) or not measures:
            raise StructuralContextError("score.measures must be non-empty")
        if not isinstance(parts, list) or not parts:
            raise StructuralContextError("score.parts must be non-empty")
        if not isinstance(events, list):
            raise StructuralContextError("score.timed_events must be a list")
        if len(parts) > self.spec.maximum_parts:
            raise StructuralContextError(
                f"score has {len(parts)} parts; maximum_parts is "
                f"{self.spec.maximum_parts}"
            )

        measure_durations = np.asarray(
            [
                _rational(
                    _mapping(measure, "measure").get("duration_ql"),
                    "measure.duration_ql",
                )
                for measure in measures
            ],
            dtype=np.float64,
        )
        if np.any(measure_durations <= 0):
            raise StructuralContextError("measure durations must be positive")
        measure_positions: dict[int, int] = {}
        for position, raw_measure in enumerate(measures):
            measure_index = _mapping(raw_measure, "measure").get("index")
            if (
                not isinstance(measure_index, int)
                or isinstance(measure_index, bool)
                or measure_index < 0
            ):
                raise StructuralContextError(
                    "measure.index must be a non-negative integer"
                )
            if measure_index in measure_positions:
                raise StructuralContextError(
                    f"duplicate measure.index: {measure_index}"
                )
            measure_positions[measure_index] = position
        part_indices = {
            _nonempty_string(
                _mapping(part, "part").get("id"), "part.id"
            ): index
            for index, part in enumerate(parts)
        }
        events_by_measure: list[list[Mapping[str, Any]]] = [
            [] for _ in measures
        ]
        for raw_event in events:
            event = _mapping(raw_event, "timed event")
            measure_index = event.get("measure_index")
            if (
                not isinstance(measure_index, int)
                or isinstance(measure_index, bool)
                or measure_index not in measure_positions
            ):
                raise StructuralContextError(
                    "timed event measure_index is absent from score.measures"
                )
            events_by_measure[measure_positions[measure_index]].append(event)

        rows = [
            self._measure_vector(
                events_by_measure[index],
                duration=float(measure_durations[index]),
                part_indices=part_indices,
            )
            for index in range(len(measures))
        ]
        return np.asarray(rows, dtype=np.float32)

    def _measure_vector(
        self,
        events: list[Mapping[str, Any]],
        *,
        duration: float,
        part_indices: Mapping[str, int],
    ) -> np.ndarray:
        pitch_classes = np.zeros(12, dtype=np.float64)
        registers = np.zeros(self.spec.register_bins, dtype=np.float64)
        onsets = np.zeros(self.spec.onset_bins, dtype=np.float64)
        duration_bins = np.zeros(
            len(self.spec.duration_ratio_boundaries) + 1,
            dtype=np.float64,
        )
        active_parts = np.zeros(self.spec.maximum_parts, dtype=np.float64)
        part_notes = np.zeros(self.spec.maximum_parts, dtype=np.float64)
        pitches: list[int] = []
        duration_ratios: list[float] = []
        rest_count = 0
        unpitched_count = 0

        for event in events:
            kind = event.get("kind")
            if kind not in {"note", "rest", "unpitched"}:
                raise StructuralContextError(f"unknown timed event kind: {kind}")
            part_id = _nonempty_string(event.get("part_id"), "event.part_id")
            if part_id not in part_indices:
                raise StructuralContextError(
                    f"event references unknown part_id: {part_id}"
                )
            part_index = part_indices[part_id]
            onset_ratio = _rational(
                event.get("measure_onset_ql"), "event.measure_onset_ql"
            ) / duration
            onset_index = min(
                self.spec.onset_bins - 1,
                max(0, int(onset_ratio * self.spec.onset_bins)),
            )
            onsets[onset_index] += 1
            event_duration_ratio = max(
                0.0,
                _rational(event.get("duration_ql"), "event.duration_ql")
                / duration,
            )
            duration_ratios.append(event_duration_ratio)
            duration_index = int(
                np.searchsorted(
                    self.spec.duration_ratio_boundaries,
                    event_duration_ratio,
                    side="right",
                )
            )
            duration_bins[duration_index] += 1
            if kind == "rest":
                rest_count += 1
                continue
            active_parts[part_index] = 1
            part_notes[part_index] += 1
            if kind == "unpitched":
                unpitched_count += 1
                continue
            midi = event.get("midi")
            if (
                not isinstance(midi, int)
                or isinstance(midi, bool)
                or not 0 <= midi <= 127
            ):
                raise StructuralContextError("note event midi must be in 0..127")
            pitches.append(midi)
            pitch_classes[midi % 12] += 1
            register_index = min(
                self.spec.register_bins - 1,
                midi * self.spec.register_bins // 128,
            )
            registers[register_index] += 1

        note_count = len(pitches)
        sounding_count = note_count + unpitched_count
        event_count = len(events)
        if note_count:
            pitch_classes /= note_count
            registers /= note_count
            pitch_values = np.asarray(pitches, dtype=np.float64)
            mean_midi = float(pitch_values.mean()) / 127.0
            midi_std = float(pitch_values.std()) / 64.0
            midi_range = float(pitch_values.max() - pitch_values.min()) / 127.0
        else:
            mean_midi = midi_std = midi_range = 0.0
        if sounding_count:
            part_notes /= sounding_count
        if event_count:
            onsets /= event_count
            duration_bins /= event_count

        scalars = np.asarray(
            [
                math.log1p(note_count),
                math.log1p(unpitched_count),
                math.log1p(rest_count),
                float(active_parts.sum()) / max(1, len(part_indices)),
                mean_midi,
                midi_std,
                midi_range,
                min(2.0, float(np.mean(duration_ratios)))
                if duration_ratios
                else 0.0,
                rest_count / max(1, event_count),
                math.log1p(duration),
            ],
            dtype=np.float64,
        )
        return np.concatenate(
            (
                pitch_classes,
                registers,
                onsets,
                duration_bins,
                active_parts,
                part_notes,
                scalars,
            )
        )


class StructuralContextDatasetBuilder:
    def __init__(
        self,
        spec: StructuralContextSpec,
        dataset: ObservationDataset,
    ):
        self.spec = spec
        self.dataset = dataset
        self.vectorizer = PartituraMeasureVectorizer(spec)

    def prepare(self) -> PreparedStructuralContextDataset:
        manifest_digest = self.dataset.manifest.get("manifest_digest")
        if manifest_digest != self.spec.expected_observation_manifest_digest:
            raise StructuralContextError(
                "observation manifest digest mismatch: "
                f"{manifest_digest} != "
                f"{self.spec.expected_observation_manifest_digest}"
            )
        if not self.dataset.manifest.get("ready"):
            raise StructuralContextError("observation dataset is not ready")
        records = self.dataset.manifest.get("records")
        if not isinstance(records, list):
            raise StructuralContextError("observation manifest records must be a list")
        self._validate_lineage_splits(records)

        vectorized_scores: list[
            tuple[
                str,
                str,
                np.ndarray,
                list[tuple[int, tuple[int, ...]]],
            ]
        ] = []
        train_measures: list[np.ndarray] = []
        score_counts: dict[str, int] = defaultdict(int)
        omitted: list[Mapping[str, Any]] = []
        for raw_record in sorted(
            records,
            key=lambda item: str(
                _mapping(item, "observation record").get("target_id")
            ),
        ):
            record = _mapping(raw_record, "observation record")
            split = _nonempty_string(record.get("split"), "record.split")
            if split not in {"train", "validation", "test"}:
                raise StructuralContextError(f"unknown record split: {split}")
            target_id = _nonempty_string(
                record.get("target_id"), "record.target_id"
            )
            relative = _safe_relative(
                record.get("observation_file"), "record.observation_file"
            )
            observation = ScoreObservation.load(
                self.dataset.root.joinpath(*relative.parts)
            )
            if observation.digest != record.get("observation_digest"):
                raise StructuralContextError(
                    f"observation digest disagrees for {target_id}"
                )
            measures = self.vectorizer.vectorize(observation)
            candidates = self._candidate_starts(len(measures))
            if not candidates:
                omitted.append(
                    {
                        "target_id": target_id,
                        "split": split,
                        "measure_count": len(measures),
                        "reason": "no nonoverlapping nonadjacent span candidate",
                    }
                )
                continue
            vectorized_scores.append(
                (target_id, split, measures, candidates)
            )
            score_counts[split] += 1
            if split == "train":
                train_measures.append(measures)

        if not train_measures:
            raise StructuralContextError("no train measures were prepared")
        feature_matrix = np.concatenate(train_measures, axis=0).astype(
            np.float64, copy=False
        )
        mean = feature_matrix.mean(axis=0).astype(np.float32)
        scale = feature_matrix.std(axis=0).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        raw_examples: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for target_id, split, measures, candidates in vectorized_scores:
            raw_examples[split].extend(
                self._examples_for_score(
                    target_id,
                    measures,
                    candidates,
                    mean=mean,
                    scale=scale,
                )
            )
        prepared_splits = {
            split: self._prepare_split(raw_examples.get(split, []), mean, scale)
            for split in ("train", "validation", "test")
        }
        for split, prepared in prepared_splits.items():
            if prepared.count == 0:
                raise StructuralContextError(
                    f"no structural-context examples in {split}"
                )
        return PreparedStructuralContextDataset(
            observation_manifest_digest=str(manifest_digest),
            feature_names=self.vectorizer.feature_names,
            feature_mean=_freeze(mean),
            feature_scale=_freeze(scale),
            splits=prepared_splits,
            score_counts=dict(sorted(score_counts.items())),
            omitted_scores=tuple(omitted),
        )

    def _examples_for_score(
        self,
        score_id: str,
        measures: np.ndarray,
        candidates: list[tuple[int, tuple[int, ...]]],
        *,
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> list[tuple[Any, ...]]:
        span = self.spec.span_measures
        candidates.sort(
            key=lambda item: self._stable_rank(score_id, item[0], "anchor")
        )
        selected = candidates[: self.spec.maximum_examples_per_score]
        normalized_measures = (measures - mean[None, :]) / scale[None, :]
        examples: list[tuple[Any, ...]] = []
        for anchor_start, negative_starts in selected:
            authentic_start = anchor_start + span
            negative_start = self._negative_start(
                score_id,
                anchor_start,
                authentic_start,
                negative_starts,
                normalized_measures,
            )
            examples.append(
                (
                    measures[anchor_start : anchor_start + span],
                    measures[authentic_start : authentic_start + span],
                    measures[negative_start : negative_start + span],
                    score_id,
                    anchor_start,
                    authentic_start,
                    negative_start,
                )
            )
        return examples

    def _candidate_starts(
        self, measure_count: int
    ) -> list[tuple[int, tuple[int, ...]]]:
        span = self.spec.span_measures
        stride = self.spec.stride_measures
        starts = range(0, measure_count - span + 1, stride)
        candidates: list[tuple[int, tuple[int, ...]]] = []
        for anchor_start in range(0, measure_count - 2 * span + 1, stride):
            authentic_start = anchor_start + span
            negative_starts = tuple(
                start
                for start in starts
                if abs(start - authentic_start)
                >= self.spec.minimum_negative_distance_measures
                and (
                    start + span <= anchor_start
                    or start >= anchor_start + 2 * span
                )
            )
            if negative_starts:
                candidates.append((anchor_start, negative_starts))
        return candidates

    def _negative_start(
        self,
        score_id: str,
        anchor_start: int,
        authentic_start: int,
        negative_starts: tuple[int, ...],
        normalized_measures: np.ndarray,
    ) -> int:
        if self.spec.negative_sampling == "random_same_score":
            index = (
                self._stable_rank(score_id, anchor_start, "negative")
                % len(negative_starts)
            )
            return negative_starts[index]

        anchor_boundary = normalized_measures[
            anchor_start + self.spec.span_measures - 1
        ]
        authentic_distance = float(
            np.mean(
                np.square(
                    anchor_boundary - normalized_measures[authentic_start]
                )
            )
        )

        def match_key(start: int) -> tuple[float, int]:
            distance = float(
                np.mean(
                    np.square(
                        anchor_boundary - normalized_measures[start]
                    )
                )
            )
            return (
                abs(distance - authentic_distance),
                self._stable_rank(score_id, start, "boundary_match"),
            )

        return min(negative_starts, key=match_key)

    def _stable_rank(self, score_id: str, start: int, purpose: str) -> int:
        payload = f"{self.spec.seed}\0{score_id}\0{start}\0{purpose}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    @staticmethod
    def _prepare_split(
        examples: list[tuple[Any, ...]],
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> PreparedContextSplit:
        if not examples:
            shape = (0, 0, len(mean))
            empty = _freeze(np.empty(shape, dtype=np.float32))
            empty_indices = _freeze(np.empty(0, dtype=np.int64))
            return PreparedContextSplit(
                anchors=empty,
                authentic=empty,
                nonadjacent=empty,
                score_ids=(),
                anchor_starts=empty_indices,
                authentic_starts=empty_indices,
                nonadjacent_starts=empty_indices,
            )

        def normalized(position: int) -> np.ndarray:
            values = np.stack([item[position] for item in examples])
            return _freeze(
                ((values - mean[None, None, :]) / scale[None, None, :]).astype(
                    np.float32
                )
            )

        return PreparedContextSplit(
            anchors=normalized(0),
            authentic=normalized(1),
            nonadjacent=normalized(2),
            score_ids=tuple(str(item[3]) for item in examples),
            anchor_starts=_freeze(
                np.asarray([item[4] for item in examples], dtype=np.int64)
            ),
            authentic_starts=_freeze(
                np.asarray([item[5] for item in examples], dtype=np.int64)
            ),
            nonadjacent_starts=_freeze(
                np.asarray([item[6] for item in examples], dtype=np.int64)
            ),
        )

    @staticmethod
    def _validate_lineage_splits(records: list[object]) -> None:
        splits_by_lineage: dict[str, set[str]] = defaultdict(set)
        for raw_record in records:
            record = _mapping(raw_record, "observation record")
            lineage = _nonempty_string(
                record.get("lineage_id"), "record.lineage_id"
            )
            split = _nonempty_string(record.get("split"), "record.split")
            splits_by_lineage[lineage].add(split)
        leaking = {
            lineage: sorted(splits)
            for lineage, splits in splits_by_lineage.items()
            if len(splits) > 1
        }
        if leaking:
            raise StructuralContextError(
                f"composition lineages leak across splits: {leaking}"
            )


class StructuralContextModel(nn.Module):
    """Learn measure features, span order, and directed adjacency jointly."""

    def __init__(self, spec: StructuralContextSpec, feature_dimension: int):
        super().__init__()
        self.baseline_residual = spec.baseline_residual
        self.measure_encoder = nn.Sequential(
            nn.Linear(feature_dimension, spec.measure_dimension),
            nn.GELU(),
            nn.LayerNorm(spec.measure_dimension),
            nn.Dropout(spec.dropout),
        )
        self.span_encoder = nn.GRU(
            input_size=spec.measure_dimension,
            hidden_size=spec.span_dimension,
            batch_first=True,
        )
        self.adjacency_scorer = nn.Sequential(
            nn.Linear(spec.span_dimension * 4, spec.score_hidden_dimension),
            nn.GELU(),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.score_hidden_dimension, 1),
        )
        if self.baseline_residual:
            output_layer = self.adjacency_scorer[-1]
            nn.init.zeros_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)

    def encode_span(self, values: torch.Tensor) -> torch.Tensor:
        measures = self.measure_encoder(values)
        _, hidden = self.span_encoder(measures)
        return hidden[-1]

    def forward(
        self, anchor: torch.Tensor, candidate: torch.Tensor
    ) -> torch.Tensor:
        anchor_span = self.encode_span(anchor)
        candidate_span = self.encode_span(candidate)
        joined = torch.cat(
            (
                anchor_span,
                candidate_span,
                torch.abs(anchor_span - candidate_span),
                anchor_span * candidate_span,
            ),
            dim=-1,
        )
        learned_score = self.adjacency_scorer(joined).squeeze(-1)
        if not self.baseline_residual:
            return learned_score
        boundary_score = -torch.mean(
            torch.square(anchor[:, -1] - candidate[:, 0]), dim=1
        )
        return boundary_score + learned_score


class StructuralContextTrainer:
    def __init__(
        self,
        spec: StructuralContextSpec,
        dataset: PreparedStructuralContextDataset,
    ):
        self.spec = spec
        self.dataset = dataset

    def train(self, output_root: str | Path) -> dict[str, Any]:
        self._seed_everything()
        model = StructuralContextModel(
            self.spec, len(self.dataset.feature_names)
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.spec.learning_rate,
            weight_decay=self.spec.weight_decay,
        )
        generator = np.random.default_rng(self.spec.seed)
        train_split = self.dataset.splits["train"]
        best_state: dict[str, torch.Tensor] | None = None
        best_validation = -math.inf
        best_step = 0
        stale_validations = 0
        history: list[dict[str, Any]] = []
        if self.spec.baseline_residual:
            initial_validation = self.evaluate(model, "validation")
            history.append(
                {
                    "step": 0,
                    "train_batch_loss": None,
                    "validation": initial_validation,
                }
            )
            best_validation = initial_validation["score_macro_accuracy"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        model.train()
        for step in range(1, self.spec.steps + 1):
            indices = generator.integers(
                0, train_split.count, size=self.spec.batch_size
            )
            anchor = torch.from_numpy(train_split.anchors[indices])
            authentic = torch.from_numpy(train_split.authentic[indices])
            nonadjacent = torch.from_numpy(train_split.nonadjacent[indices])
            optimizer.zero_grad(set_to_none=True)
            authentic_scores = model(anchor, authentic)
            nonadjacent_scores = model(anchor, nonadjacent)
            loss = torch_functional.softplus(
                -(authentic_scores - nonadjacent_scores)
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.spec.gradient_clip
            )
            optimizer.step()

            if step % self.spec.validation_interval:
                continue
            validation = self.evaluate(model, "validation")
            history.append(
                {
                    "step": step,
                    "train_batch_loss": float(loss.detach()),
                    "validation": validation,
                }
            )
            score = validation["score_macro_accuracy"]
            if score > best_validation:
                best_validation = score
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                stale_validations = 0
            else:
                stale_validations += 1
            model.train()
            if stale_validations >= self.spec.patience:
                break

        if best_state is None:
            raise StructuralContextError("training produced no selected checkpoint")
        model.load_state_dict(best_state)
        validation = self.evaluate(model, "validation")
        test = self.evaluate(model, "test") if self.spec.evaluate_test else None
        baseline = {
            split: self.evaluate_boundary_baseline(split)
            for split in ("validation", "test")
            if split != "test" or self.spec.evaluate_test
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": self.spec.experiment_id,
            "experiment_spec_digest": self.spec.digest,
            "observation_manifest_digest": (
                self.dataset.observation_manifest_digest
            ),
            "task": {
                "positive": "authentic immediately following span",
                "negative": (
                    "nonoverlapping nonadjacent span from the same score"
                ),
                "span_measures": self.spec.span_measures,
                "negative_sampling": self.spec.negative_sampling,
            },
            "dataset": {
                "feature_count": len(self.dataset.feature_names),
                "score_counts": dict(self.dataset.score_counts),
                "example_counts": {
                    split: prepared.count
                    for split, prepared in self.dataset.splits.items()
                },
                "omitted_scores": list(self.dataset.omitted_scores),
            },
            "model": {
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "measure_dimension": self.spec.measure_dimension,
                "span_dimension": self.spec.span_dimension,
                "baseline_residual": self.spec.baseline_residual,
            },
            "training": {
                "best_step": best_step,
                "steps_completed": history[-1]["step"],
                "history": history,
            },
            "baseline": baseline,
            "validation": validation,
            "test": test,
        }
        report["validation_delta_over_baseline"] = (
            validation["score_macro_accuracy"]
            - baseline["validation"]["score_macro_accuracy"]
        )
        if test is not None:
            report["test_delta_over_baseline"] = (
                test["score_macro_accuracy"]
                - baseline["test"]["score_macro_accuracy"]
            )
        output = Path(output_root)
        output.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output / "checkpoint.pt"
        self._write_checkpoint(checkpoint_path, model, report)
        report["checkpoint_digest"] = self._file_digest(checkpoint_path)
        report_payload = dict(report)
        report_payload["report_digest"] = _canonical_digest(report_payload)
        self._write_json(output / "report.json", report_payload)
        return report_payload

    def evaluate(
        self, model: StructuralContextModel, split_name: str
    ) -> dict[str, Any]:
        split = self.dataset.splits[split_name]
        gaps: list[np.ndarray] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, split.count, self.spec.evaluation_batch_size):
                stop = min(split.count, start + self.spec.evaluation_batch_size)
                anchor = torch.from_numpy(split.anchors[start:stop].copy())
                authentic = torch.from_numpy(
                    split.authentic[start:stop].copy()
                )
                nonadjacent = torch.from_numpy(
                    split.nonadjacent[start:stop].copy()
                )
                gaps.append(
                    (
                        model(anchor, authentic)
                        - model(anchor, nonadjacent)
                    )
                    .cpu()
                    .numpy()
                )
        return self._metrics(np.concatenate(gaps), split.score_ids)

    def evaluate_boundary_baseline(self, split_name: str) -> dict[str, Any]:
        split = self.dataset.splits[split_name]
        authentic_distance = np.mean(
            np.square(split.anchors[:, -1] - split.authentic[:, 0]), axis=1
        )
        nonadjacent_distance = np.mean(
            np.square(split.anchors[:, -1] - split.nonadjacent[:, 0]), axis=1
        )
        return self._metrics(
            nonadjacent_distance - authentic_distance,
            split.score_ids,
        )

    @staticmethod
    def _metrics(
        gaps: np.ndarray, score_ids: tuple[str, ...]
    ) -> dict[str, Any]:
        correct = (gaps > 0).astype(np.float64)
        ties = gaps == 0
        correct[ties] = 0.5
        by_score: dict[str, list[float]] = defaultdict(list)
        for score_id, value in zip(score_ids, correct, strict=True):
            by_score[score_id].append(float(value))
        return {
            "example_count": len(gaps),
            "pair_accuracy": float(correct.mean()),
            "score_macro_accuracy": float(
                np.mean([np.mean(values) for values in by_score.values()])
            ),
            "mean_score_gap": float(gaps.mean()),
        }

    def _seed_everything(self) -> None:
        random.seed(self.spec.seed)
        np.random.seed(self.spec.seed)
        torch.manual_seed(self.spec.seed)
        torch.use_deterministic_algorithms(True)

    def _write_checkpoint(
        self,
        path: Path,
        model: StructuralContextModel,
        report: Mapping[str, Any],
    ) -> None:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": self.dataset.feature_names,
                    "feature_mean": self.dataset.feature_mean.copy(),
                    "feature_scale": self.dataset.feature_scale.copy(),
                    "model_config": {
                        "measure_dimension": self.spec.measure_dimension,
                        "span_dimension": self.spec.span_dimension,
                        "score_hidden_dimension": (
                            self.spec.score_hidden_dimension
                        ),
                        "dropout": self.spec.dropout,
                        "baseline_residual": self.spec.baseline_residual,
                    },
                    "span_measures": self.spec.span_measures,
                    "experiment_spec_digest": report[
                        "experiment_spec_digest"
                    ],
                    "observation_manifest_digest": report[
                        "observation_manifest_digest"
                    ],
                },
                temporary_path,
            )
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
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

    @staticmethod
    def _file_digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"


def evaluate_structural_context_checkpoint(
    *,
    spec: StructuralContextSpec,
    checkpoint_path: str | Path,
    training_dataset: ObservationDataset,
    holdout_dataset: ObservationDataset,
) -> dict[str, Any]:
    """Evaluate one selected checkpoint without fitting to the holdout."""

    checkpoint_file = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise StructuralContextError(
            f"cannot load structural-context checkpoint {checkpoint_file}: "
            f"{error}"
        ) from error
    checkpoint_data = _mapping(checkpoint, "structural-context checkpoint")
    if checkpoint_data.get("experiment_spec_digest") != spec.digest:
        raise StructuralContextError(
            "checkpoint experiment digest does not match evaluation spec"
        )
    if (
        checkpoint_data.get("observation_manifest_digest")
        != training_dataset.manifest.get("manifest_digest")
    ):
        raise StructuralContextError(
            "checkpoint training manifest digest does not match"
        )

    training_records = training_dataset.manifest.get("records")
    holdout_records = holdout_dataset.manifest.get("records")
    if not isinstance(training_records, list) or not isinstance(
        holdout_records, list
    ):
        raise StructuralContextError("evaluation manifests require records")
    training_source_digests = {
        _nonempty_string(
            _mapping(record, "training record").get("source_digest"),
            "training record.source_digest",
        )
        for record in training_records
    }
    holdout_source_digests = {
        _nonempty_string(
            _mapping(record, "holdout record").get("source_digest"),
            "holdout record.source_digest",
        )
        for record in holdout_records
    }
    source_overlap = sorted(
        training_source_digests & holdout_source_digests
    )
    if source_overlap:
        raise StructuralContextError(
            f"external holdout repeats training source digests: {source_overlap}"
        )
    training_lineages = {
        _nonempty_string(
            _mapping(record, "training record").get("lineage_id"),
            "training record.lineage_id",
        )
        for record in training_records
    }
    holdout_lineages = {
        _nonempty_string(
            _mapping(record, "holdout record").get("lineage_id"),
            "holdout record.lineage_id",
        )
        for record in holdout_records
    }
    lineage_overlap = sorted(training_lineages & holdout_lineages)
    if lineage_overlap:
        raise StructuralContextError(
            f"external holdout repeats training lineages: {lineage_overlap}"
        )

    builder = StructuralContextDatasetBuilder(spec, holdout_dataset)
    checkpoint_feature_names = tuple(
        checkpoint_data.get("feature_names", ())
    )
    if checkpoint_feature_names != builder.vectorizer.feature_names:
        raise StructuralContextError(
            "checkpoint feature vocabulary does not match the vectorizer"
        )
    feature_mean = np.asarray(
        checkpoint_data.get("feature_mean"), dtype=np.float32
    )
    feature_scale = np.asarray(
        checkpoint_data.get("feature_scale"), dtype=np.float32
    )
    feature_count = len(builder.vectorizer.feature_names)
    if feature_mean.shape != (feature_count,) or feature_scale.shape != (
        feature_count,
    ):
        raise StructuralContextError(
            "checkpoint normalization has the wrong feature dimension"
        )
    if np.any(feature_scale <= 0) or not np.all(
        np.isfinite(feature_mean)
    ) or not np.all(np.isfinite(feature_scale)):
        raise StructuralContextError(
            "checkpoint normalization must be finite with positive scales"
        )

    examples: list[tuple[Any, ...]] = []
    omitted: list[Mapping[str, Any]] = []
    score_ids: set[str] = set()
    for raw_record in sorted(
        holdout_records,
        key=lambda item: str(
            _mapping(item, "holdout record").get("target_id")
        ),
    ):
        record = _mapping(raw_record, "holdout record")
        if record.get("split") != "test":
            raise StructuralContextError(
                "external holdout records must use the test split"
            )
        target_id = _nonempty_string(
            record.get("target_id"), "holdout record.target_id"
        )
        relative = _safe_relative(
            record.get("observation_file"),
            "holdout record.observation_file",
        )
        observation = ScoreObservation.load(
            holdout_dataset.root.joinpath(*relative.parts)
        )
        if observation.digest != record.get("observation_digest"):
            raise StructuralContextError(
                f"holdout observation digest disagrees for {target_id}"
            )
        measures = builder.vectorizer.vectorize(observation)
        candidates = builder._candidate_starts(len(measures))
        if not candidates:
            omitted.append(
                {
                    "target_id": target_id,
                    "measure_count": len(measures),
                    "reason": "no nonoverlapping nonadjacent span candidate",
                }
            )
            continue
        examples.extend(
            builder._examples_for_score(
                target_id,
                measures,
                candidates,
                mean=feature_mean,
                scale=feature_scale,
            )
        )
        score_ids.add(target_id)
    prepared_split = builder._prepare_split(
        examples, feature_mean, feature_scale
    )
    if prepared_split.count == 0:
        raise StructuralContextError(
            "external holdout produced no structural-context examples"
        )
    prepared = PreparedStructuralContextDataset(
        observation_manifest_digest=str(
            holdout_dataset.manifest.get("manifest_digest")
        ),
        feature_names=builder.vectorizer.feature_names,
        feature_mean=_freeze(feature_mean),
        feature_scale=_freeze(feature_scale),
        splits={"external_holdout": prepared_split},
        score_counts={"external_holdout": len(score_ids)},
        omitted_scores=tuple(omitted),
    )
    model = StructuralContextModel(spec, feature_count)
    state = checkpoint_data.get("model_state")
    if not isinstance(state, Mapping):
        raise StructuralContextError("checkpoint model_state must be an object")
    try:
        model.load_state_dict(state)
    except RuntimeError as error:
        raise StructuralContextError(
            f"checkpoint model state is incompatible: {error}"
        ) from error
    trainer = StructuralContextTrainer(spec, prepared)
    model_metrics = trainer.evaluate(model, "external_holdout")
    baseline_metrics = trainer.evaluate_boundary_baseline(
        "external_holdout"
    )
    return {
        "schema_version": 1,
        "experiment_id": spec.experiment_id,
        "experiment_spec_digest": spec.digest,
        "checkpoint_digest": trainer._file_digest(checkpoint_file),
        "training_manifest_digest": training_dataset.manifest.get(
            "manifest_digest"
        ),
        "holdout_manifest_digest": holdout_dataset.manifest.get(
            "manifest_digest"
        ),
        "leakage_checks": {
            "source_digest_overlap": source_overlap,
            "lineage_overlap": lineage_overlap,
        },
        "holdout": {
            "score_count": len(score_ids),
            "lineage_count": len(holdout_lineages),
            "example_count": prepared_split.count,
            "omitted_scores": omitted,
        },
        "baseline": baseline_metrics,
        "model": model_metrics,
        "model_delta_over_baseline": (
            model_metrics["score_macro_accuracy"]
            - baseline_metrics["score_macro_accuracy"]
        ),
    }


class StructuralSeamScorer:
    """Apply the selected continuation model to one Partitura observation."""

    def __init__(
        self,
        *,
        spec: StructuralContextSpec,
        model: StructuralContextModel,
        feature_names: tuple[str, ...],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
    ):
        self.spec = spec
        self.model = model
        self.feature_names = feature_names
        self.feature_mean = _freeze(
            np.asarray(feature_mean, dtype=np.float32)
        )
        self.feature_scale = _freeze(
            np.asarray(feature_scale, dtype=np.float32)
        )
        self.vectorizer = PartituraMeasureVectorizer(spec)

    @classmethod
    def load(
        cls,
        *,
        spec: StructuralContextSpec,
        checkpoint_path: str | Path,
    ) -> StructuralSeamScorer:
        path = Path(checkpoint_path)
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise StructuralContextError(
                f"cannot load structural-context checkpoint {path}: {error}"
            ) from error
        data = _mapping(checkpoint, "structural-context checkpoint")
        if data.get("experiment_spec_digest") != spec.digest:
            raise StructuralContextError(
                "checkpoint experiment digest does not match scorer spec"
            )
        vectorizer = PartituraMeasureVectorizer(spec)
        feature_names = tuple(data.get("feature_names", ()))
        if feature_names != vectorizer.feature_names:
            raise StructuralContextError(
                "checkpoint feature vocabulary does not match scorer"
            )
        feature_mean = np.asarray(
            data.get("feature_mean"), dtype=np.float32
        )
        feature_scale = np.asarray(
            data.get("feature_scale"), dtype=np.float32
        )
        dimension = len(feature_names)
        if feature_mean.shape != (dimension,) or feature_scale.shape != (
            dimension,
        ):
            raise StructuralContextError(
                "checkpoint normalization has the wrong scorer dimension"
            )
        if np.any(feature_scale <= 0):
            raise StructuralContextError(
                "checkpoint scorer scales must be positive"
            )
        model = StructuralContextModel(spec, dimension)
        state = data.get("model_state")
        if not isinstance(state, Mapping):
            raise StructuralContextError(
                "checkpoint model_state must be an object"
            )
        try:
            model.load_state_dict(state)
        except RuntimeError as error:
            raise StructuralContextError(
                f"checkpoint model state is incompatible: {error}"
            ) from error
        model.eval()
        return cls(
            spec=spec,
            model=model,
            feature_names=feature_names,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
        )

    def score(self, observation: ScoreObservation) -> StructuralSeamSignal:
        measures = self.vectorizer.vectorize(observation)
        span = self.spec.span_measures
        starts = tuple(
            range(
                0,
                len(measures) - 2 * span + 1,
                self.spec.stride_measures,
            )
        )
        if not starts:
            raise StructuralContextError(
                f"score needs at least {2 * span} measures for seam scoring"
            )
        normalized = (
            measures - self.feature_mean[None, :]
        ) / self.feature_scale[None, :]
        anchors = np.stack(
            [normalized[start : start + span] for start in starts]
        ).astype(np.float32)
        successors = np.stack(
            [
                normalized[start + span : start + 2 * span]
                for start in starts
            ]
        ).astype(np.float32)
        with torch.no_grad():
            learned = (
                self.model(
                    torch.from_numpy(anchors),
                    torch.from_numpy(successors),
                )
                .cpu()
                .numpy()
            )
        boundary = -np.mean(
            np.square(anchors[:, -1] - successors[:, 0]), axis=1
        )
        worst = int(np.argmin(learned))
        return StructuralSeamSignal(
            observation_digest=observation.digest,
            adjacency_count=len(starts),
            learned_mean=float(np.mean(learned)),
            learned_tenth_percentile=float(
                np.percentile(learned, 10, method="linear")
            ),
            learned_minimum=float(learned[worst]),
            boundary_mean=float(np.mean(boundary)),
            residual_mean=float(np.mean(learned - boundary)),
            worst_successor_start_position=starts[worst] + span,
        )


class StructuralSeamCritic:
    """Expose seam signals for Ruby-exported workflow candidates."""

    def __init__(self, scorer: StructuralSeamScorer):
        self.scorer = scorer

    @classmethod
    def load(
        cls,
        *,
        spec: StructuralContextSpec,
        checkpoint_path: str | Path,
    ) -> StructuralSeamCritic:
        return cls(
            StructuralSeamScorer.load(
                spec=spec,
                checkpoint_path=checkpoint_path,
            )
        )

    def evaluate(
        self, request: SelectionRequest
    ) -> tuple[LearnedCriticResult, ...]:
        results: list[LearnedCriticResult] = []
        observations = request.to_dict()["candidate_observations"]
        for assessment in request.assessments:
            candidate = _mapping(
                assessment.get("candidate"),
                "selection assessment candidate",
            )
            candidate_id = _nonempty_string(
                candidate.get("candidate_id"),
                "selection candidate id",
            )
            raw_observation = observations.get(candidate_id)
            if raw_observation is None:
                continue
            signal = self.scorer.score(
                ScoreObservation.from_dict(raw_observation)
            )
            results.append(
                LearnedCriticResult(
                    critic="structural-context-v4",
                    scale="seam",
                    target_path=_nonempty_string(
                        candidate.get("target_path"),
                        "selection candidate target_path",
                    ),
                    candidate_id=candidate_id,
                    findings=(
                        "Comparative continuation signal; no calibrated "
                        "acceptance threshold.",
                    ),
                    features={
                        "adjacency_count": float(signal.adjacency_count),
                        "boundary_mean": signal.boundary_mean,
                        "learned_mean": signal.learned_mean,
                        "learned_minimum": signal.learned_minimum,
                        "learned_tenth_percentile": (
                            signal.learned_tenth_percentile
                        ),
                        "residual_mean": signal.residual_mean,
                        "worst_successor_start_position": float(
                            signal.worst_successor_start_position
                        ),
                    },
                    score=signal.learned_tenth_percentile,
                )
            )
        return tuple(results)
