"""Deterministic multi-task learning over Partitura-owned score-span features."""

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

from generation.composition.annotation_dataset import (
    AnnotationDataset,
    TrainingExample,
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

_SPLITS = ("train", "validation", "test")
_SELECTION_METRIC = "mean_target_macro_f1"
_SAMPLING_STRATEGIES = {"uniform_example", "uniform_label"}


class ScoreSpanEncoderError(ValueError):
    """Raised when an encoder experiment or artifact violates its contract."""


def _number(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ScoreSpanEncoderError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScoreSpanEncoderError(f"{label} must be finite")
    if minimum is not None and result < minimum:
        raise ScoreSpanEncoderError(f"{label} must be at least {minimum}")
    if maximum is not None and result >= maximum:
        raise ScoreSpanEncoderError(f"{label} must be less than {maximum}")
    return result


@dataclass(frozen=True)
class ScoreSpanEncoderSpec:
    experiment_id: str
    annotation_manifest: PurePosixPath
    expected_annotation_manifest_digest: str
    baseline_report: PurePosixPath
    expected_baseline_report_digest: str
    output_root: PurePosixPath
    targets: tuple[str, ...]
    seed: int
    embedding_dim: int
    representation_dim: int
    dropout: float
    steps: int
    batch_size_per_target: int
    learning_rate: float
    weight_decay: float
    validation_interval: int
    patience: int
    gradient_clip: float
    evaluation_batch_size: int
    selection_metric: str
    sampling_strategy: str
    evaluate_test: bool
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> ScoreSpanEncoderSpec:
        spec_path = Path(path)
        try:
            value = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ScoreSpanEncoderError(
                f"cannot read score-span encoder spec {spec_path}: {error}"
            ) from error
        return cls.from_dict(value)

    @classmethod
    def from_dict(cls, value: object) -> ScoreSpanEncoderSpec:
        data = _mapping(value, "score-span encoder spec")
        if data.get("schema_version") != 1:
            raise ScoreSpanEncoderError(
                "score-span encoder spec schema_version must be 1"
            )
        model = _mapping(data.get("model"), "score-span encoder model")
        training = _mapping(data.get("training"), "score-span encoder training")
        targets = tuple(
            _identifier(item, "score-span encoder target")
            for item in _list(data.get("targets"), "score-span encoder targets")
        )
        if not targets or len(targets) != len(set(targets)):
            raise ScoreSpanEncoderError(
                "score-span encoder targets must be non-empty and unique"
            )
        annotation_manifest = _safe_relative(
            data.get("annotation_manifest"), "annotation_manifest"
        )
        baseline_report = _safe_relative(
            data.get("baseline_report"), "baseline_report"
        )
        output_root = _safe_relative(data.get("output_root"), "output_root")
        for label, path in (
            ("annotation_manifest", annotation_manifest),
            ("baseline_report", baseline_report),
            ("output_root", output_root),
        ):
            if path.parts[0] != "outputs":
                raise ScoreSpanEncoderError(f"{label} must stay under outputs/")
        selection_metric = _string(
            data.get("selection_metric"), "selection_metric"
        )
        if selection_metric != _SELECTION_METRIC:
            raise ScoreSpanEncoderError(
                f"selection_metric must be {_SELECTION_METRIC}"
            )
        sampling_strategy = _string(
            training.get("sampling_strategy", "uniform_example"),
            "training.sampling_strategy",
        )
        if sampling_strategy not in _SAMPLING_STRATEGIES:
            raise ScoreSpanEncoderError(
                "training.sampling_strategy must be uniform_example or uniform_label"
            )
        evaluate_test = training.get("evaluate_test", True)
        if not isinstance(evaluate_test, bool):
            raise ScoreSpanEncoderError("training.evaluate_test must be boolean")
        seed = _positive_integer(data.get("seed"), "seed", allow_zero=True)
        dropout = _number(model.get("dropout"), "model.dropout", minimum=0, maximum=1)
        learning_rate = _number(
            training.get("learning_rate"),
            "training.learning_rate",
            minimum=0,
        )
        weight_decay = _number(
            training.get("weight_decay"),
            "training.weight_decay",
            minimum=0,
        )
        gradient_clip = _number(
            training.get("gradient_clip"),
            "training.gradient_clip",
            minimum=0,
        )
        if learning_rate == 0 or gradient_clip == 0:
            raise ScoreSpanEncoderError(
                "learning_rate and gradient_clip must be positive"
            )
        validation_interval = _positive_integer(
            training.get("validation_interval"),
            "training.validation_interval",
        )
        steps = _positive_integer(training.get("steps"), "training.steps")
        if validation_interval > steps:
            raise ScoreSpanEncoderError(
                "validation_interval cannot exceed training steps"
            )
        return cls(
            experiment_id=_identifier(data.get("experiment_id"), "experiment_id"),
            annotation_manifest=annotation_manifest,
            expected_annotation_manifest_digest=_string(
                data.get("expected_annotation_manifest_digest"),
                "expected_annotation_manifest_digest",
            ),
            baseline_report=baseline_report,
            expected_baseline_report_digest=_string(
                data.get("expected_baseline_report_digest"),
                "expected_baseline_report_digest",
            ),
            output_root=output_root,
            targets=targets,
            seed=seed,
            embedding_dim=_positive_integer(
                model.get("embedding_dim"), "model.embedding_dim"
            ),
            representation_dim=_positive_integer(
                model.get("representation_dim"), "model.representation_dim"
            ),
            dropout=dropout,
            steps=steps,
            batch_size_per_target=_positive_integer(
                training.get("batch_size_per_target"),
                "training.batch_size_per_target",
            ),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            validation_interval=validation_interval,
            patience=_positive_integer(
                training.get("patience"), "training.patience"
            ),
            gradient_clip=gradient_clip,
            evaluation_batch_size=_positive_integer(
                training.get("evaluation_batch_size"),
                "training.evaluation_batch_size",
            ),
            selection_metric=selection_metric,
            sampling_strategy=sampling_strategy,
            evaluate_test=evaluate_test,
            raw=json.loads(json.dumps(data)),
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.raw)


@dataclass(frozen=True)
class PreparedSplit:
    values: np.ndarray
    label_indices: np.ndarray
    labels: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class PreparedTarget:
    id: str
    feature_names: tuple[str, ...]
    feature_ids: np.ndarray
    label_vocabulary: tuple[str, ...]
    splits: Mapping[str, PreparedSplit]


@dataclass(frozen=True)
class PreparedScoreSpanDataset:
    annotation_manifest_digest: str
    feature_vocabulary: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    targets: Mapping[str, PreparedTarget]
    lineage_leakage_checks: tuple[Mapping[str, Any], ...]


class ScoreSpanDatasetPreparer:
    def __init__(
        self,
        spec: ScoreSpanEncoderSpec,
        dataset: AnnotationDataset,
    ):
        self.spec = spec
        self.dataset = dataset

    def prepare(self) -> PreparedScoreSpanDataset:
        manifest_digest = str(self.dataset.manifest.get("manifest_digest"))
        if manifest_digest != self.spec.expected_annotation_manifest_digest:
            raise ScoreSpanEncoderError(
                "annotation manifest digest does not match encoder experiment"
            )
        if not self.dataset.manifest.get("ready"):
            raise ScoreSpanEncoderError("annotation dataset is not ready")
        leakage = self._lineage_leakage_checks()
        if not all(check["passed"] for check in leakage):
            raise ScoreSpanEncoderError(
                "composition lineages leak across encoder splits"
            )
        self._validate_targets()
        grouped: dict[str, list[TrainingExample]] = defaultdict(list)
        for example in self.dataset.examples():
            if example.target in self.spec.targets:
                grouped[example.target].append(example)
        schemas = self._feature_schemas(grouped)
        feature_vocabulary = tuple(
            sorted({name for schema in schemas.values() for name in schema})
        )
        feature_index = {
            feature_name: index
            for index, feature_name in enumerate(feature_vocabulary)
        }
        feature_mean, feature_scale = self._normalizer(
            grouped,
            schemas,
            feature_index,
        )
        targets = {
            target_id: self._prepare_target(
                target_id,
                grouped[target_id],
                schemas[target_id],
                feature_index,
                feature_mean,
                feature_scale,
            )
            for target_id in self.spec.targets
        }
        return PreparedScoreSpanDataset(
            annotation_manifest_digest=manifest_digest,
            feature_vocabulary=feature_vocabulary,
            feature_mean=_immutable_array(feature_mean),
            feature_scale=_immutable_array(feature_scale),
            targets=MappingProxyType(targets),
            lineage_leakage_checks=tuple(leakage),
        )

    def _validate_targets(self) -> None:
        available = {
            str(target["id"]): str(target["availability"])
            for target in self.dataset.manifest.get("targets", [])
        }
        missing = set(self.spec.targets) - set(available)
        unsupported = {
            target
            for target in self.spec.targets
            if available.get(target) != "supported"
        }
        if missing or unsupported:
            raise ScoreSpanEncoderError(
                "encoder targets are absent or unavailable: "
                f"missing={sorted(missing)}, unavailable={sorted(unsupported)}"
            )

    def _feature_schemas(
        self,
        grouped: Mapping[str, Sequence[TrainingExample]],
    ) -> dict[str, tuple[str, ...]]:
        schemas: dict[str, tuple[str, ...]] = {}
        for target_id in self.spec.targets:
            examples = grouped.get(target_id, ())
            by_split = {
                split: [example for example in examples if example.split == split]
                for split in _SPLITS
            }
            if any(not items for items in by_split.values()):
                raise ScoreSpanEncoderError(
                    f"encoder target {target_id} lacks a required split"
                )
            target_schemas = {example.feature_names for example in examples}
            if len(target_schemas) != 1:
                raise ScoreSpanEncoderError(
                    f"feature schema varies for encoder target {target_id}"
                )
            schemas[target_id] = next(iter(target_schemas))
        return schemas

    def _normalizer(
        self,
        grouped: Mapping[str, Sequence[TrainingExample]],
        schemas: Mapping[str, tuple[str, ...]],
        feature_index: Mapping[str, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        size = len(feature_index)
        counts = np.zeros(size, dtype=np.int64)
        sums = np.zeros(size, dtype=np.float64)
        squared_sums = np.zeros(size, dtype=np.float64)
        for target_id in self.spec.targets:
            train = [
                example
                for example in grouped[target_id]
                if example.split == "train"
            ]
            matrix = np.asarray(
                [example.features for example in train],
                dtype=np.float64,
            )
            for column, feature_name in enumerate(schemas[target_id]):
                index = feature_index[feature_name]
                values = matrix[:, column]
                counts[index] += len(values)
                sums[index] += values.sum()
                squared_sums[index] += np.square(values).sum()
        if np.any(counts == 0):
            raise ScoreSpanEncoderError(
                "feature normalizer encountered a feature without training values"
            )
        mean = sums / counts
        variance = np.maximum((squared_sums / counts) - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale == 0.0] = 1.0
        return mean.astype(np.float32), scale.astype(np.float32)

    def _prepare_target(
        self,
        target_id: str,
        examples: Sequence[TrainingExample],
        feature_names: tuple[str, ...],
        feature_index: Mapping[str, int],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
    ) -> PreparedTarget:
        train_labels = tuple(
            sorted(
                {
                    example.label
                    for example in examples
                    if example.split == "train"
                }
            )
        )
        label_index = {label: index for index, label in enumerate(train_labels)}
        feature_ids = np.asarray(
            [feature_index[name] for name in feature_names],
            dtype=np.int64,
        )
        splits = {}
        for split in _SPLITS:
            items = [example for example in examples if example.split == split]
            values = np.asarray(
                [example.features for example in items],
                dtype=np.float32,
            )
            values = (
                values - feature_mean[feature_ids][None, :]
            ) / feature_scale[feature_ids][None, :]
            indices = np.asarray(
                [label_index.get(example.label, -1) for example in items],
                dtype=np.int64,
            )
            splits[split] = PreparedSplit(
                values=_immutable_array(values),
                label_indices=_immutable_array(indices),
                labels=tuple(example.label for example in items),
            )
        return PreparedTarget(
            id=target_id,
            feature_names=feature_names,
            feature_ids=_immutable_array(feature_ids),
            label_vocabulary=train_labels,
            splits=MappingProxyType(splits),
        )

    def _lineage_leakage_checks(self) -> list[dict[str, Any]]:
        by_split: dict[str, set[str]] = defaultdict(set)
        for record in self.dataset.manifest.get("records", []):
            by_split[str(record["split"])].add(str(record["lineage_id"]))
        checks = []
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = sorted(by_split[left] & by_split[right])
            checks.append(
                {
                    "left": left,
                    "right": right,
                    "overlap": overlap,
                    "passed": not overlap,
                }
            )
        return checks


class ScoreSpanEncoder(nn.Module):
    """Learned set encoder shared by target-specific classification heads."""

    def __init__(
        self,
        *,
        feature_count: int,
        label_counts: Mapping[str, int],
        embedding_dim: int,
        representation_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.feature_embedding = nn.Embedding(feature_count, embedding_dim)
        self.value_projection = nn.Linear(1, embedding_dim)
        self.token_normalization = nn.LayerNorm(embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
        )
        self.heads = nn.ModuleDict(
            {
                target_id: nn.Linear(representation_dim, label_count)
                for target_id, label_count in label_counts.items()
            }
        )

    def forward(
        self,
        *,
        target_id: str,
        feature_ids: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        names = self.feature_embedding(feature_ids).unsqueeze(0)
        scalars = self.value_projection(values.unsqueeze(-1))
        tokens = torch_functional.gelu(
            self.token_normalization(names + scalars)
        )
        representation = self.encoder(tokens.mean(dim=1))
        return self.heads[target_id](representation), representation


class ScoreSpanEncoderTrainer:
    def __init__(
        self,
        spec: ScoreSpanEncoderSpec,
        dataset: PreparedScoreSpanDataset,
    ):
        self.spec = spec
        self.dataset = dataset

    def train(
        self,
        *,
        output_root: str | Path,
        baseline_report: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._validate_baseline_report(baseline_report)
        _set_determinism(self.spec.seed)
        model = self._model()
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
            training_loss = self._training_step(
                model,
                optimizer,
                generator,
            )
            completed_steps = step
            if not self._should_validate(step):
                continue
            validation = self._evaluate(model, "validation")
            selection_score = _mean_target_macro_f1(validation)
            history.append(
                {
                    "step": step,
                    "training_loss": round(training_loss, 8),
                    "selection_score": round(selection_score, 8),
                    "target_macro_f1": {
                        target_id: metrics["macro_f1"]
                        for target_id, metrics in validation.items()
                    },
                }
            )
            if selection_score > best_score:
                best_score = selection_score
                best_step = step
                best_validation = validation
                best_state = _clone_state(model.state_dict())
                without_improvement = 0
            else:
                without_improvement += 1
            if without_improvement >= self.spec.patience:
                break
        if best_state is None or best_validation is None:
            raise ScoreSpanEncoderError(
                "training completed without a validation checkpoint"
            )
        model.load_state_dict(best_state)
        test_metrics = (
            self._evaluate(model, "test")
            if self.spec.evaluate_test
            else None
        )
        model_digest = _model_digest(best_state)
        output = Path(output_root)
        checkpoint_path = output / "checkpoint.pt"
        report_path = output / "report.json"
        checkpoint = self._checkpoint_payload(best_state, model_digest, best_step)
        _write_torch(checkpoint_path, checkpoint)
        report = self._report(
            baseline_report=baseline_report,
            best_validation=best_validation,
            test_metrics=test_metrics,
            history=history,
            best_step=best_step,
            completed_steps=completed_steps,
            model_digest=model_digest,
            checkpoint_digest=_file_digest(checkpoint_path),
        )
        _write_json(report_path, report)
        return report

    def _model(self) -> ScoreSpanEncoder:
        return ScoreSpanEncoder(
            feature_count=len(self.dataset.feature_vocabulary),
            label_counts={
                target_id: len(target.label_vocabulary)
                for target_id, target in self.dataset.targets.items()
            },
            embedding_dim=self.spec.embedding_dim,
            representation_dim=self.spec.representation_dim,
            dropout=self.spec.dropout,
        )

    def _training_step(
        self,
        model: ScoreSpanEncoder,
        optimizer: torch.optim.Optimizer,
        generator: np.random.Generator,
    ) -> float:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for target_id in self.spec.targets:
            target = self.dataset.targets[target_id]
            split = target.splits["train"]
            indices = self._training_indices(split, generator)
            logits, _ = model(
                target_id=target_id,
                feature_ids=torch.tensor(
                    np.asarray(target.feature_ids)
                ),
                values=torch.tensor(
                    np.asarray(split.values[indices])
                ),
            )
            labels = torch.tensor(
                np.asarray(split.label_indices[indices])
            )
            losses.append(torch_functional.cross_entropy(logits, labels))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.spec.gradient_clip,
        )
        optimizer.step()
        return float(loss.detach())

    def _training_indices(
        self,
        split: PreparedSplit,
        generator: np.random.Generator,
    ) -> np.ndarray:
        if self.spec.sampling_strategy == "uniform_example":
            return generator.integers(
                0,
                split.count,
                size=self.spec.batch_size_per_target,
            )
        labels = np.unique(split.label_indices)
        if np.any(labels < 0):
            raise ScoreSpanEncoderError(
                "training split contains a label outside its vocabulary"
            )
        selected_labels = generator.choice(
            labels,
            size=self.spec.batch_size_per_target,
            replace=True,
        )
        return np.asarray(
            [
                generator.choice(np.flatnonzero(split.label_indices == label))
                for label in selected_labels
            ],
            dtype=np.int64,
        )

    def _should_validate(self, step: int) -> bool:
        return (
            step % self.spec.validation_interval == 0
            or step == self.spec.steps
        )

    def _evaluate(
        self,
        model: ScoreSpanEncoder,
        split_name: str,
    ) -> dict[str, dict[str, Any]]:
        model.eval()
        results = {}
        with torch.inference_mode():
            for target_id in self.spec.targets:
                target = self.dataset.targets[target_id]
                split = target.splits[split_name]
                predicted: list[str] = []
                for start in range(0, split.count, self.spec.evaluation_batch_size):
                    end = min(start + self.spec.evaluation_batch_size, split.count)
                    logits, _ = model(
                        target_id=target_id,
                        feature_ids=torch.tensor(
                            np.asarray(target.feature_ids)
                        ),
                        values=torch.tensor(
                            np.asarray(split.values[start:end])
                        ),
                    )
                    predicted.extend(
                        target.label_vocabulary[index]
                        for index in logits.argmax(dim=1).tolist()
                    )
                results[target_id] = _classification_metrics(
                    split.labels,
                    predicted,
                )
        return results

    def _validate_baseline_report(
        self,
        baseline_report: Mapping[str, Any],
    ) -> None:
        if (
            baseline_report.get("baseline_report_digest")
            != self.spec.expected_baseline_report_digest
        ):
            raise ScoreSpanEncoderError(
                "baseline report digest does not match encoder experiment"
            )
        if (
            baseline_report.get("annotation_manifest_digest")
            != self.dataset.annotation_manifest_digest
        ):
            raise ScoreSpanEncoderError(
                "baseline report belongs to a different annotation manifest"
            )

    def _checkpoint_payload(
        self,
        state: Mapping[str, torch.Tensor],
        model_digest: str,
        best_step: int,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "experiment_id": self.spec.experiment_id,
            "spec_digest": self.spec.digest,
            "annotation_manifest_digest": self.dataset.annotation_manifest_digest,
            "model_digest": model_digest,
            "best_step": best_step,
            "feature_vocabulary": self.dataset.feature_vocabulary,
            "feature_mean": torch.tensor(
                np.asarray(self.dataset.feature_mean)
            ),
            "feature_scale": torch.tensor(
                np.asarray(self.dataset.feature_scale)
            ),
            "target_feature_names": {
                target_id: target.feature_names
                for target_id, target in self.dataset.targets.items()
            },
            "target_feature_ids": {
                target_id: torch.tensor(
                    np.asarray(target.feature_ids)
                )
                for target_id, target in self.dataset.targets.items()
            },
            "target_label_vocabulary": {
                target_id: target.label_vocabulary
                for target_id, target in self.dataset.targets.items()
            },
            "model": {
                "embedding_dim": self.spec.embedding_dim,
                "representation_dim": self.spec.representation_dim,
                "dropout": self.spec.dropout,
            },
            "state_dict": dict(state),
        }

    def _report(
        self,
        *,
        baseline_report: Mapping[str, Any],
        best_validation: Mapping[str, Any],
        test_metrics: Mapping[str, Any] | None,
        history: Sequence[Mapping[str, Any]],
        best_step: int,
        completed_steps: int,
        model_digest: str,
        checkpoint_digest: str,
    ) -> dict[str, Any]:
        baseline_targets = {
            str(target["target"]): target
            for target in baseline_report["targets"]
            if target["availability"] == "supported"
        }
        targets = []
        for target_id in self.spec.targets:
            prepared = self.dataset.targets[target_id]
            baseline = baseline_targets[target_id]["models"]["nearest_centroid"]
            target_report: dict[str, Any] = {
                "target": target_id,
                "feature_count": len(prepared.feature_names),
                "train_label_count": len(prepared.label_vocabulary),
                "example_counts": {
                    split: prepared.splits[split].count
                    for split in _SPLITS
                },
                "unseen_evaluation_labels": {
                    split: sorted(
                        set(prepared.splits[split].labels)
                        - set(prepared.label_vocabulary)
                    )
                    for split in ("validation", "test")
                },
                "validation": best_validation[target_id],
                "nearest_centroid_validation": baseline["validation"],
                "validation_macro_f1_delta": round(
                    best_validation[target_id]["macro_f1"]
                    - baseline["validation"]["macro_f1"],
                    6,
                ),
            }
            if test_metrics is not None:
                model_test = test_metrics[target_id]
                baseline_test = baseline["test"]
                target_report.update(
                    {
                        "test": model_test,
                        "nearest_centroid_test": baseline_test,
                        "test_macro_f1_delta": round(
                            model_test["macro_f1"] - baseline_test["macro_f1"],
                            6,
                        ),
                    }
                )
            targets.append(target_report)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": self.spec.experiment_id,
            "spec_digest": self.spec.digest,
            "annotation_manifest_digest": self.dataset.annotation_manifest_digest,
            "baseline_report_digest": self.spec.expected_baseline_report_digest,
            "selection_metric": self.spec.selection_metric,
            "lineage_leakage_checks": list(
                self.dataset.lineage_leakage_checks
            ),
            "model_digest": model_digest,
            "checkpoint_digest": checkpoint_digest,
            "feature_vocabulary_size": len(
                self.dataset.feature_vocabulary
            ),
            "training": {
                "best_step": best_step,
                "completed_steps": completed_steps,
                "sampling_strategy": self.spec.sampling_strategy,
                "evaluate_test": self.spec.evaluate_test,
                "validation_history": list(history),
            },
            "best_validation_score": round(
                _mean_target_macro_f1(best_validation),
                8,
            ),
            "final_test_score": (
                round(_mean_target_macro_f1(test_metrics), 8)
                if test_metrics is not None
                else None
            ),
            "targets": targets,
        }
        payload["report_digest"] = _canonical_digest(payload)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        return payload


def load_baseline_report(path: str | Path) -> Mapping[str, Any]:
    report_path = Path(path)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScoreSpanEncoderError(
            f"cannot read baseline report {report_path}: {error}"
        ) from error
    data = _mapping(report, "baseline report")
    claimed = _string(data.get("baseline_report_digest"), "baseline_report_digest")
    payload = dict(data)
    payload.pop("baseline_report_digest", None)
    if _canonical_digest(payload) != claimed:
        raise ScoreSpanEncoderError("baseline report digest mismatch")
    return dict(data)


def verify_encoder_artifacts(
    *,
    spec: ScoreSpanEncoderSpec,
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
        raise ScoreSpanEncoderError(
            f"cannot read encoder artifacts: {error}"
        ) from error
    checkpoint_data = _mapping(checkpoint, "encoder checkpoint")
    report_data = _mapping(report, "encoder report")
    claimed_report = _string(report_data.get("report_digest"), "report_digest")
    report_payload = dict(report_data)
    report_payload.pop("report_digest", None)
    report_payload.pop("generated_at", None)
    if _canonical_digest(report_payload) != claimed_report:
        raise ScoreSpanEncoderError("encoder report digest mismatch")
    state = _mapping(checkpoint_data.get("state_dict"), "encoder state_dict")
    actual_model_digest = _model_digest(state)
    expected = {
        "spec_digest": spec.digest,
        "annotation_manifest_digest": spec.expected_annotation_manifest_digest,
        "model_digest": actual_model_digest,
    }
    for key, value in expected.items():
        if checkpoint_data.get(key) != value or report_data.get(key) != value:
            raise ScoreSpanEncoderError(
                f"encoder artifact lineage mismatch for {key}"
            )
    actual_checkpoint_digest = _file_digest(checkpoint_file)
    if report_data.get("checkpoint_digest") != actual_checkpoint_digest:
        raise ScoreSpanEncoderError("encoder checkpoint file digest mismatch")
    return {
        "status": "ok",
        "experiment_id": spec.experiment_id,
        "model_digest": actual_model_digest,
        "report_digest": claimed_report,
        "best_step": checkpoint_data.get("best_step"),
    }


def _classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, Any]:
    if len(expected) != len(predicted) or not expected:
        raise ScoreSpanEncoderError("encoder predictions do not align")
    labels = sorted(set(expected) | set(predicted))
    correct = sum(left == right for left, right in zip(expected, predicted))
    f1_values = []
    for label in labels:
        true_positive = sum(
            left == label and right == label
            for left, right in zip(expected, predicted)
        )
        false_positive = sum(
            left != label and right == label
            for left, right in zip(expected, predicted)
        )
        false_negative = sum(
            left == label and right != label
            for left, right in zip(expected, predicted)
        )
        denominator = (2 * true_positive) + false_positive + false_negative
        f1_values.append(
            0.0 if denominator == 0 else (2 * true_positive) / denominator
        )
    return {
        "example_count": len(expected),
        "accuracy": round(correct / len(expected), 6),
        "macro_f1": round(sum(f1_values) / len(f1_values), 6),
    }


def _mean_target_macro_f1(metrics: Mapping[str, Any]) -> float:
    return sum(
        float(target_metrics["macro_f1"])
        for target_metrics in metrics.values()
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
            raise ScoreSpanEncoderError(
                f"encoder state value is not a tensor: {name}"
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)
