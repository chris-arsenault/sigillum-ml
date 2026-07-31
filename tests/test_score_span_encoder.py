import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from generation.composition.annotation_dataset import TrainingExample
from generation.composition.score_span_encoder import (
    PreparedSplit,
    ScoreSpanDatasetPreparer,
    ScoreSpanEncoderError,
    ScoreSpanEncoderSpec,
    ScoreSpanEncoderTrainer,
    verify_encoder_artifacts,
)


class _FixtureDataset:
    def __init__(self, *, leak: bool = False):
        validation_lineage = "lineage_train" if leak else "lineage_validation"
        self.manifest = {
            "manifest_digest": "sha256:fixture-annotation-manifest",
            "ready": True,
            "targets": [
                {"id": "relation", "availability": "supported"},
                {"id": "boundary", "availability": "supported"},
            ],
            "records": [
                {
                    "split": "train",
                    "lineage_id": "lineage_train",
                },
                {
                    "split": "validation",
                    "lineage_id": validation_lineage,
                },
                {
                    "split": "test",
                    "lineage_id": "lineage_test",
                },
            ],
        }
        self._examples = tuple(self._make_examples())

    def examples(self):
        return self._examples

    def _make_examples(self):
        examples = []
        for target, names in (
            ("relation", ("shared", "relation_only")),
            ("boundary", ("shared",)),
        ):
            for split, values in (
                ("train", (-2.0, -1.0, 1.0, 2.0)),
                ("validation", (-1.5, -0.5, 0.5, 1.5)),
                ("test", (-1.75, -0.75, 0.75, 1.75)),
            ):
                for index, value in enumerate(values):
                    features = (
                        (value, value * 0.5)
                        if target == "relation"
                        else (value,)
                    )
                    examples.append(
                        TrainingExample(
                            id=f"{target}:{split}:{index}",
                            target=target,
                            label="negative" if value < 0 else "positive",
                            feature_names=names,
                            features=features,
                            target_id=f"score_{split}",
                            lineage_id=f"lineage_{split}",
                            split=split,
                            scope={},
                            provenance={},
                            metadata={},
                        )
                    )
        return examples


class ScoreSpanEncoderTests(unittest.TestCase):
    def test_preparation_uses_train_only_statistics_and_rejects_leakage(self):
        spec = self._spec()
        prepared = ScoreSpanDatasetPreparer(
            spec,
            _FixtureDataset(),
        ).prepare()

        shared = prepared.feature_vocabulary.index("shared")
        self.assertEqual(0.0, prepared.feature_mean[shared])
        self.assertTrue(prepared.feature_scale[shared] > 0)
        self.assertTrue(
            all(check["passed"] for check in prepared.lineage_leakage_checks)
        )
        with self.assertRaises(ValueError):
            prepared.feature_mean[shared] = 4.0

        with self.assertRaisesRegex(
            ScoreSpanEncoderError,
            "composition lineages leak",
        ):
            ScoreSpanDatasetPreparer(
                spec,
                _FixtureDataset(leak=True),
            ).prepare()

    def test_training_is_deterministic_and_artifacts_verify(self):
        spec = self._spec()
        baseline = {
            "baseline_report_digest": "sha256:fixture-baseline",
            "annotation_manifest_digest": (
                "sha256:fixture-annotation-manifest"
            ),
            "targets": [
                self._baseline_target("relation"),
                self._baseline_target("boundary"),
            ],
        }
        model_digests = []
        result_metrics = []
        with tempfile.TemporaryDirectory(
            prefix="sigillum-score-span-encoder-"
        ) as temporary:
            root = Path(temporary)
            for run in ("one", "two"):
                prepared = ScoreSpanDatasetPreparer(
                    spec,
                    _FixtureDataset(),
                ).prepare()
                output = root / run
                report = ScoreSpanEncoderTrainer(spec, prepared).train(
                    output_root=output,
                    baseline_report=baseline,
                )
                verified = verify_encoder_artifacts(
                    spec=spec,
                    checkpoint_path=output / "checkpoint.pt",
                    report_path=output / "report.json",
                )
                self.assertEqual("ok", verified["status"])
                self.assertEqual(report["model_digest"], verified["model_digest"])
                model_digests.append(report["model_digest"])
                result_metrics.append(
                    [
                        (target["validation"], target["test"])
                        for target in report["targets"]
                    ]
                )

        self.assertEqual(model_digests[0], model_digests[1])
        self.assertEqual(result_metrics[0], result_metrics[1])

    def test_artifact_verifier_rejects_report_tampering(self):
        spec = self._spec()
        prepared = ScoreSpanDatasetPreparer(
            spec,
            _FixtureDataset(),
        ).prepare()
        baseline = {
            "baseline_report_digest": "sha256:fixture-baseline",
            "annotation_manifest_digest": (
                "sha256:fixture-annotation-manifest"
            ),
            "targets": [
                self._baseline_target("relation"),
                self._baseline_target("boundary"),
            ],
        }
        with tempfile.TemporaryDirectory(
            prefix="sigillum-score-span-encoder-"
        ) as temporary:
            output = Path(temporary)
            ScoreSpanEncoderTrainer(spec, prepared).train(
                output_root=output,
                baseline_report=baseline,
            )
            report_path = output / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["best_validation_score"] += 0.1
            report_path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(
                ScoreSpanEncoderError,
                "report digest mismatch",
            ):
                verify_encoder_artifacts(
                    spec=spec,
                    checkpoint_path=output / "checkpoint.pt",
                    report_path=report_path,
                )

    def test_label_balanced_sampling_and_validation_only_mode(self):
        value = self._spec_dict()
        value["training"]["sampling_strategy"] = "uniform_label"
        value["training"]["evaluate_test"] = False
        spec = ScoreSpanEncoderSpec.from_dict(value)
        prepared = ScoreSpanDatasetPreparer(
            spec,
            _FixtureDataset(),
        ).prepare()
        trainer = ScoreSpanEncoderTrainer(spec, prepared)
        split = PreparedSplit(
            values=np.zeros((10, 1), dtype=np.float32),
            label_indices=np.asarray([0] * 9 + [1], dtype=np.int64),
            labels=tuple(["majority"] * 9 + ["minority"]),
        )
        indices = trainer._training_indices(
            split,
            np.random.default_rng(14),
        )

        self.assertIn(1, split.label_indices[indices])
        with tempfile.TemporaryDirectory(
            prefix="sigillum-score-span-validation-only-"
        ) as temporary:
            report = trainer.train(
                output_root=temporary,
                baseline_report={
                    "baseline_report_digest": "sha256:fixture-baseline",
                    "annotation_manifest_digest": (
                        "sha256:fixture-annotation-manifest"
                    ),
                    "targets": [
                        self._baseline_target("relation"),
                        self._baseline_target("boundary"),
                    ],
                },
            )

        self.assertIsNone(report["final_test_score"])
        self.assertFalse(report["training"]["evaluate_test"])
        self.assertTrue(all("test" not in item for item in report["targets"]))

    @staticmethod
    def _spec():
        return ScoreSpanEncoderSpec.from_dict(
            ScoreSpanEncoderTests._spec_dict()
        )

    @staticmethod
    def _spec_dict():
        return {
                "schema_version": 1,
                "experiment_id": "fixture_score_span_encoder",
                "annotation_manifest": "outputs/annotations/manifest.json",
                "expected_annotation_manifest_digest": (
                    "sha256:fixture-annotation-manifest"
                ),
                "baseline_report": "outputs/annotations/baselines.json",
                "expected_baseline_report_digest": (
                    "sha256:fixture-baseline"
                ),
                "output_root": "outputs/encoder",
                "targets": ["relation", "boundary"],
                "seed": 42,
                "model": {
                    "embedding_dim": 8,
                    "representation_dim": 8,
                    "dropout": 0.0,
                },
                "training": {
                    "steps": 6,
                    "batch_size_per_target": 4,
                    "learning_rate": 0.01,
                    "weight_decay": 0.0,
                    "validation_interval": 2,
                    "patience": 3,
                    "gradient_clip": 1.0,
                    "evaluation_batch_size": 8,
                },
                "selection_metric": "mean_target_macro_f1",
            }

    @staticmethod
    def _baseline_target(target_id):
        metrics = {
            "example_count": 4,
            "accuracy": 0.5,
            "macro_f1": 0.333333,
        }
        return {
            "target": target_id,
            "availability": "supported",
            "models": {
                "nearest_centroid": {
                    "validation": metrics,
                    "test": metrics,
                }
            },
        }


if __name__ == "__main__":
    unittest.main()
