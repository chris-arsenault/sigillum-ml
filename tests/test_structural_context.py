import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from generation.composition.observation_dataset import (
    ObservationDataset,
    ScoreObservation,
    _canonical_digest,
)
from generation.composition.structural_context import (
    PreparedContextSplit,
    PreparedStructuralContextDataset,
    StructuralContextDatasetBuilder,
    StructuralContextError,
    StructuralContextModel,
    StructuralContextSpec,
    StructuralContextTrainer,
    StructuralSeamCritic,
    StructuralSeamScorer,
    evaluate_structural_context_checkpoint,
)
from generation.composition.protocol import SelectionRequest


class StructuralContextTests(unittest.TestCase):
    def test_real_task_builder_uses_same_score_nonadjacent_spans(self):
        spec = self._spec()
        with tempfile.TemporaryDirectory(
            prefix="sigillum-structural-context-"
        ) as temporary:
            dataset = self._dataset(Path(temporary))
            prepared = StructuralContextDatasetBuilder(spec, dataset).prepare()

        self.assertEqual(3, sum(prepared.score_counts.values()))
        self.assertEqual(142, len(prepared.feature_names))
        for split in prepared.splits.values():
            self.assertGreater(split.count, 0)
            self.assertTrue(
                np.all(
                    split.authentic_starts
                    == split.anchor_starts + spec.span_measures
                )
            )
            self.assertTrue(
                np.all(
                    np.abs(
                        split.nonadjacent_starts - split.authentic_starts
                    )
                    >= spec.minimum_negative_distance_measures
                )
            )
            self.assertFalse(split.anchors.flags.writeable)
            self.assertTrue(
                all(score_id.startswith("score_") for score_id in split.score_ids)
            )

    def test_builder_rejects_lineage_leakage(self):
        spec = self._spec()
        with tempfile.TemporaryDirectory(
            prefix="sigillum-structural-context-leak-"
        ) as temporary:
            dataset = self._dataset(Path(temporary), leak=True)
            with self.assertRaisesRegex(
                StructuralContextError, "lineages leak"
            ):
                StructuralContextDatasetBuilder(spec, dataset).prepare()

    def test_hierarchical_model_learns_directed_span_context(self):
        spec_value = self._spec_dict()
        spec_value["task"]["span_measures"] = 2
        spec_value["model"] = {
            "measure_dimension": 12,
            "span_dimension": 12,
            "score_hidden_dimension": 12,
            "dropout": 0.0,
        }
        spec_value["training"].update(
            {
                "steps": 80,
                "batch_size": 16,
                "validation_interval": 10,
                "patience": 8,
                "evaluation_batch_size": 64,
            }
        )
        spec = StructuralContextSpec.from_dict(spec_value)
        prepared = self._synthetic_prepared()
        with tempfile.TemporaryDirectory(
            prefix="sigillum-structural-context-train-"
        ) as temporary:
            report = StructuralContextTrainer(spec, prepared).train(temporary)
            output = Path(temporary)
            self.assertTrue((output / "checkpoint.pt").is_file())
            self.assertTrue((output / "report.json").is_file())
            checkpoint = torch.load(
                output / "checkpoint.pt",
                map_location="cpu",
                weights_only=False,
            )

        self.assertGreaterEqual(
            report["validation"]["score_macro_accuracy"], 0.9
        )
        self.assertGreaterEqual(report["test"]["score_macro_accuracy"], 0.9)
        self.assertGreater(report["validation_delta_over_baseline"], 0.0)
        self.assertEqual(prepared.feature_names, checkpoint["feature_names"])
        np.testing.assert_array_equal(
            prepared.feature_mean, checkpoint["feature_mean"]
        )
        np.testing.assert_array_equal(
            prepared.feature_scale, checkpoint["feature_scale"]
        )

    def test_residual_model_starts_at_exact_boundary_baseline(self):
        value = self._spec_dict()
        value["model"]["baseline_residual"] = True
        spec = StructuralContextSpec.from_dict(value)
        model = StructuralContextModel(spec, feature_dimension=3)
        anchor = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, -1.0, 0.5],
                    [2.0, 0.0, -0.5],
                    [3.0, 1.0, 0.0],
                ]
            ]
        )
        candidate = torch.tensor(
            [
                [
                    [4.0, 0.0, 0.5],
                    [5.0, -1.0, 0.0],
                    [6.0, 0.0, -0.5],
                    [7.0, 1.0, 0.0],
                ]
            ]
        )

        with torch.no_grad():
            actual = model(anchor, candidate)
        expected = -torch.mean(
            torch.square(anchor[:, -1] - candidate[:, 0]), dim=1
        )

        torch.testing.assert_close(actual, expected)

    def test_boundary_matched_negative_controls_for_local_distance(self):
        value = self._spec_dict()
        value["task"]["negative_sampling"] = (
            "boundary_matched_same_score"
        )
        spec = StructuralContextSpec.from_dict(value)
        builder = StructuralContextDatasetBuilder(
            spec,
            ObservationDataset(root=Path("."), manifest={}),
        )
        measures = np.zeros((24, 1), dtype=np.float32)
        measures[4, 0] = 10.0
        measures[12, 0] = 9.0
        measures[16, 0] = 14.0

        selected = builder._negative_start(
            "score_fixture",
            anchor_start=0,
            authentic_start=4,
            negative_starts=(12, 16),
            normalized_measures=measures,
        )

        self.assertEqual(12, selected)

    def test_external_evaluation_reuses_checkpoint_without_score_leakage(self):
        spec = self._spec()
        with tempfile.TemporaryDirectory(
            prefix="sigillum-structural-context-external-"
        ) as temporary:
            root = Path(temporary)
            training = self._dataset(root / "training")
            holdout = self._external_dataset(root / "holdout")
            builder = StructuralContextDatasetBuilder(spec, training)
            feature_names = builder.vectorizer.feature_names
            model = StructuralContextModel(spec, len(feature_names))
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_names": feature_names,
                    "feature_mean": np.zeros(
                        len(feature_names), dtype=np.float32
                    ),
                    "feature_scale": np.ones(
                        len(feature_names), dtype=np.float32
                    ),
                    "experiment_spec_digest": spec.digest,
                    "observation_manifest_digest": (
                        training.manifest["manifest_digest"]
                    ),
                },
                checkpoint_path,
            )

            report = evaluate_structural_context_checkpoint(
                spec=spec,
                checkpoint_path=checkpoint_path,
                training_dataset=training,
                holdout_dataset=holdout,
            )
            scorer = StructuralSeamScorer.load(
                spec=spec,
                checkpoint_path=checkpoint_path,
            )
            signal = scorer.score(
                ScoreObservation.load(
                    holdout.root / "score_external.json"
                )
            )
            observation = ScoreObservation.load(
                holdout.root / "score_external.json"
            )
            candidate_id = "candidate:external"
            critic_results = StructuralSeamCritic(scorer).evaluate(
                SelectionRequest(
                    request_id="selection:external",
                    proposal_request_id="proposal:external",
                    snapshot={"snapshot_digest": "opaque"},
                    action={"action_id": "action:opaque"},
                    original_candidate_id="original",
                    assessments=(
                        {
                            "candidate": {
                                "candidate_id": candidate_id,
                                "target_path": "span:external",
                            },
                            "critic_results": (),
                        },
                    ),
                    candidate_observations={
                        candidate_id: observation.to_dict()
                    },
                )
            )
            overlapping = self._external_dataset(
                root / "overlap", source_digit="1"
            )
            with self.assertRaisesRegex(
                StructuralContextError, "repeats training source digests"
            ):
                evaluate_structural_context_checkpoint(
                    spec=spec,
                    checkpoint_path=checkpoint_path,
                    training_dataset=training,
                    holdout_dataset=overlapping,
                )

        self.assertEqual(1, report["holdout"]["score_count"])
        self.assertGreater(report["holdout"]["example_count"], 0)
        self.assertEqual(
            [], report["leakage_checks"]["source_digest_overlap"]
        )
        self.assertGreater(signal.adjacency_count, 0)
        self.assertTrue(np.isfinite(signal.learned_mean))
        self.assertEqual(
            holdout.manifest["records"][0]["observation_digest"],
            signal.observation_digest,
        )
        self.assertEqual(1, len(critic_results))
        self.assertEqual("seam", critic_results[0].scale)
        self.assertIsNone(critic_results[0].passed)
        self.assertIsNone(critic_results[0].confidence)
        self.assertIn(
            "learned_tenth_percentile", critic_results[0].features
        )

    @staticmethod
    def _spec() -> StructuralContextSpec:
        return StructuralContextSpec.from_dict(
            StructuralContextTests._spec_dict()
        )

    @staticmethod
    def _spec_dict() -> dict:
        return {
            "schema_version": 1,
            "experiment_id": "fixture_structural_context",
            "observation_manifest": "outputs/dataset/manifest.json",
            "expected_observation_manifest_digest": (
                "sha256:fixture-observation-manifest"
            ),
            "output_root": "outputs/experiment",
            "seed": 42,
            "task": {
                "span_measures": 4,
                "stride_measures": 4,
                "minimum_negative_distance_measures": 12,
                "maximum_examples_per_score": 6,
            },
            "vectorizer": {
                "maximum_parts": 48,
                "onset_bins": 8,
                "register_bins": 8,
                "duration_ratio_boundaries": [
                    0.03125,
                    0.0625,
                    0.125,
                    0.25,
                    0.5,
                    1.0,
                    2.0,
                ],
            },
            "model": {
                "measure_dimension": 16,
                "span_dimension": 16,
                "score_hidden_dimension": 16,
                "dropout": 0.0,
            },
            "training": {
                "steps": 20,
                "batch_size": 8,
                "learning_rate": 0.005,
                "weight_decay": 0.0,
                "validation_interval": 5,
                "patience": 4,
                "gradient_clip": 1.0,
                "evaluation_batch_size": 64,
                "evaluate_test": True,
            },
            "selection_metric": "validation_score_macro_accuracy",
            "baseline": "boundary_profile_distance",
        }

    @staticmethod
    def _dataset(root: Path, *, leak: bool = False) -> ObservationDataset:
        root.mkdir(parents=True, exist_ok=True)
        records = []
        for split, offset in (
            ("train", 0),
            ("validation", 7),
            ("test", 13),
        ):
            target_id = f"score_{split}"
            observation = StructuralContextTests._observation(offset)
            path = root / f"{target_id}.json"
            path.write_text(
                json.dumps(observation, sort_keys=True), encoding="utf-8"
            )
            lineage = "lineage_train" if leak and split == "validation" else (
                f"lineage_{split}"
            )
            records.append(
                {
                    "target_id": target_id,
                    "lineage_id": lineage,
                    "split": split,
                    "observation_file": path.name,
                    "observation_digest": observation["observation_digest"],
                    "source_digest": observation["source"]["source_digest"],
                }
            )
        return ObservationDataset(
            root=root,
            manifest={
                "manifest_digest": "sha256:fixture-observation-manifest",
                "ready": True,
                "records": records,
            },
        )

    @staticmethod
    def _external_dataset(
        root: Path, *, source_digit: str = "3"
    ) -> ObservationDataset:
        root.mkdir(parents=True, exist_ok=True)
        observation = StructuralContextTests._observation(
            19, source_digit=source_digit
        )
        path = root / "score_external.json"
        path.write_text(
            json.dumps(observation, sort_keys=True), encoding="utf-8"
        )
        return ObservationDataset(
            root=root,
            manifest={
                "manifest_digest": "sha256:fixture-external-manifest",
                "ready": True,
                "records": [
                    {
                        "target_id": "score_external",
                        "lineage_id": "lineage_external",
                        "split": "test",
                        "observation_file": path.name,
                        "observation_digest": observation[
                            "observation_digest"
                        ],
                        "source_digest": observation["source"][
                            "source_digest"
                        ],
                    }
                ],
            },
        )

    @staticmethod
    def _observation(offset: int, *, source_digit: str = "1") -> dict:
        measures = [
            {
                "index": index,
                "number": str(index + 1),
                "offset_ql": f"{index * 4}/1",
                "duration_ql": "4/1",
                "implicit": False,
            }
            for index in range(40)
        ]
        events = []
        for measure_index in range(40):
            for part_index, part_id in enumerate(("P1", "P2")):
                events.append(
                    {
                        "kind": "note",
                        "midi": 48
                        + (measure_index + offset + part_index * 5) % 24,
                        "written_midi": 48
                        + (measure_index + offset + part_index * 5) % 24,
                        "part_id": part_id,
                        "measure_index": measure_index,
                        "measure_number": str(measure_index + 1),
                        "onset_ql": f"{measure_index * 4}/1",
                        "measure_onset_ql": "0/1",
                        "duration_ql": "1/1",
                        "staff": 1,
                        "voice": "1",
                        "chord": False,
                        "cue": False,
                        "grace": False,
                        "ties": [],
                        "event_id": (
                            f"event_{measure_index}_{part_index}"
                        ),
                    }
                )
        payload = {
            "schema_version": 1,
            "source": {
                "source_digest": "sha256:" + source_digit * 64,
                "document_digest": "sha256:" + "2" * 64,
            },
            "score": {
                "creators": [],
                "key_events": [],
                "measures": measures,
                "meter_events": [],
                "parts": [
                    {
                        "id": "P1",
                        "name": "Part 1",
                        "abbreviation": "P1",
                    },
                    {
                        "id": "P2",
                        "name": "Part 2",
                        "abbreviation": "P2",
                    },
                ],
                "tempo_events": [],
                "timed_events": events,
                "title": "Fixture",
            },
            "summary": {
                "part_count": 2,
                "measure_count": 40,
                "event_count": len(events),
                "pitched_note_count": len(events),
                "warning_count": 0,
            },
            "warnings": [],
        }
        payload["observation_digest"] = _canonical_digest(payload)
        return payload

    @staticmethod
    def _synthetic_prepared() -> PreparedStructuralContextDataset:
        def split(offset: float) -> PreparedContextSplit:
            anchors = []
            authentic = []
            nonadjacent = []
            score_ids = []
            for index in range(32):
                base = ((index % 8) - 4) / 4 + offset
                direction = 1.0 if index % 2 else -1.0
                anchors.append(
                    [[base, direction], [base + direction, direction]]
                )
                authentic.append(
                    [
                        [base + 2 * direction, direction],
                        [base + 3 * direction, direction],
                    ]
                )
                nonadjacent.append(
                    [
                        [base + 2 * direction, direction],
                        [base + direction, -direction],
                    ]
                )
                score_ids.append(f"score_{index % 4}")
            count = len(score_ids)
            return PreparedContextSplit(
                anchors=np.asarray(anchors, dtype=np.float32),
                authentic=np.asarray(authentic, dtype=np.float32),
                nonadjacent=np.asarray(nonadjacent, dtype=np.float32),
                score_ids=tuple(score_ids),
                anchor_starts=np.arange(count, dtype=np.int64),
                authentic_starts=np.arange(count, dtype=np.int64) + 2,
                nonadjacent_starts=np.arange(count, dtype=np.int64) + 8,
            )

        return PreparedStructuralContextDataset(
            observation_manifest_digest=(
                "sha256:fixture-observation-manifest"
            ),
            feature_names=("position", "direction"),
            feature_mean=np.zeros(2, dtype=np.float32),
            feature_scale=np.ones(2, dtype=np.float32),
            splits={
                "train": split(0.0),
                "validation": split(0.1),
                "test": split(-0.1),
            },
            score_counts={"train": 4, "validation": 4, "test": 4},
            omitted_scores=(),
        )


if __name__ == "__main__":
    unittest.main()
