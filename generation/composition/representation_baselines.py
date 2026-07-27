"""Deterministic majority and feature-centroid baselines for annotation targets."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from generation.composition.annotation_dataset import (
    AnnotationDataset,
    AnnotationDatasetError,
    AnnotationDatasetSpec,
    TrainingExample,
)
from generation.composition.observation_dataset import _canonical_digest


@dataclass(frozen=True)
class RepresentationBaselineRunner:
    spec: AnnotationDatasetSpec
    dataset: AnnotationDataset

    def run(self) -> Mapping[str, Any]:
        leakage = self._lineage_leakage()
        if not all(check["passed"] for check in leakage):
            raise AnnotationDatasetError("composition lineages leak across dataset splits")
        examples_by_target: dict[str, list[TrainingExample]] = defaultdict(list)
        for example in self.dataset.examples():
            examples_by_target[example.target].append(example)
        results = []
        for target in self.spec.targets:
            examples = examples_by_target.get(target.id, [])
            if target.availability == "unavailable":
                results.append(self._unavailable_result(target.id, target.reason))
            else:
                results.append(self._supported_result(target.id, examples))
        payload: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": self.spec.dataset_id,
            "annotation_manifest_digest": self.dataset.manifest["manifest_digest"],
            "lineage_leakage_checks": leakage,
            "targets": results,
        }
        payload["baseline_report_digest"] = _canonical_digest(payload)
        return payload

    def _supported_result(
        self, target_id: str, examples: Sequence[TrainingExample]
    ) -> dict[str, Any]:
        by_split = {
            split: [example for example in examples if example.split == split]
            for split in ("train", "validation", "test")
        }
        if any(not split_examples for split_examples in by_split.values()):
            raise AnnotationDatasetError(
                f"supported target {target_id} lacks train, validation, or test examples"
            )
        train = by_split["train"]
        labels = sorted({example.label for example in examples})
        schemas = {example.feature_names for example in examples}
        if len(schemas) != 1:
            raise AnnotationDatasetError(
                f"feature schema varies for supported target {target_id}"
            )
        majority_label = self._majority_label(train)
        mean, scale, centroids = self._centroids(train)
        models = {
            "majority": self._evaluate_model(
                by_split,
                lambda items: [majority_label] * len(items),
            ),
            "nearest_centroid": self._evaluate_model(
                by_split,
                lambda items: self._centroid_predictions(
                    items, mean, scale, centroids
                ),
            ),
        }
        train_labels = {example.label for example in train}
        return {
            "target": target_id,
            "availability": "supported",
            "example_counts": {
                split: len(split_examples)
                for split, split_examples in by_split.items()
            },
            "label_count": len(labels),
            "feature_count": len(next(iter(schemas))),
            "unseen_evaluation_labels": {
                split: sorted(
                    {example.label for example in by_split[split]} - train_labels
                )
                for split in ("validation", "test")
            },
            "models": models,
        }

    @staticmethod
    def _unavailable_result(target_id: str, reason: str | None) -> dict[str, Any]:
        return {
            "target": target_id,
            "availability": "unavailable",
            "example_counts": {"train": 0, "validation": 0, "test": 0},
            "metric": None,
            "baselines": [],
            "reason": reason,
        }

    @staticmethod
    def _majority_label(examples: Sequence[TrainingExample]) -> str:
        counts = Counter(example.label for example in examples)
        maximum = max(counts.values())
        return min(label for label, count in counts.items() if count == maximum)

    @staticmethod
    def _centroids(
        examples: Sequence[TrainingExample],
    ) -> tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]:
        matrix = np.asarray([example.features for example in examples], dtype=float)
        mean = matrix.mean(axis=0)
        scale = matrix.std(axis=0)
        scale[scale == 0.0] = 1.0
        standardized = (matrix - mean) / scale
        labels = np.asarray([example.label for example in examples])
        centroids = {
            label: standardized[labels == label].mean(axis=0)
            for label in sorted(set(labels))
        }
        return mean, scale, centroids

    @staticmethod
    def _centroid_predictions(
        examples: Sequence[TrainingExample],
        mean: np.ndarray,
        scale: np.ndarray,
        centroids: Mapping[str, np.ndarray],
    ) -> list[str]:
        matrix = np.asarray([example.features for example in examples], dtype=float)
        standardized = (matrix - mean) / scale
        labels = sorted(centroids)
        centroid_matrix = np.stack([centroids[label] for label in labels])
        distances = ((standardized[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(
            axis=2
        )
        return [labels[index] for index in distances.argmin(axis=1)]

    def _evaluate_model(
        self,
        by_split: Mapping[str, Sequence[TrainingExample]],
        predict: Any,
    ) -> dict[str, Any]:
        return {
            split: self._metrics(
                [example.label for example in by_split[split]],
                predict(by_split[split]),
            )
            for split in ("validation", "test")
        }

    @staticmethod
    def _metrics(expected: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
        if len(expected) != len(predicted) or not expected:
            raise AnnotationDatasetError("baseline predictions do not align")
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
            f1_values.append(0.0 if denominator == 0 else (2 * true_positive) / denominator)
        return {
            "example_count": len(expected),
            "accuracy": round(correct / len(expected), 6),
            "macro_f1": round(sum(f1_values) / len(f1_values), 6),
        }

    def _lineage_leakage(self) -> list[dict[str, Any]]:
        by_split: dict[str, set[str]] = defaultdict(set)
        for raw_record in self.dataset.manifest["records"]:
            by_split[str(raw_record["split"])].add(str(raw_record["lineage_id"]))
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
