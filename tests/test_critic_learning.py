import hashlib
import tempfile
import unittest
from pathlib import Path

from generation.composition.critic_learning import (
    CriticCorpusIndex,
    CriticCorpusPreparer,
    CriticLearningError,
    CriticLearningSpec,
    PairwiseCriticTrainer,
    build_critic_corpus_index,
    load_critic_pairs,
    verify_critic_artifacts,
)
from generation.composition.evidence import (
    HumanPreferenceRecord,
    PairwiseExample,
    PairwiseReviewRecord,
    TrajectoryRecord,
)


CRITERIA = ("coherence", "identity", "seams", "orchestration", "reserve")


class CriticLearningTests(unittest.TestCase):
    def test_preparation_is_criterion_specific_and_rejects_run_leakage(self):
        spec = self._spec()
        pairs = self._pairs(spec)
        prepared = CriticCorpusPreparer(
            spec,
            pairs,
            corpus_index_digest="sha256:fixture-corpus",
        ).prepare()

        self.assertTrue(prepared.audit["ready"])
        self.assertTrue(prepared.feature_vocabulary)
        self.assertTrue(
            all(item["ready"] for item in prepared.audit["criteria"])
        )
        with self.assertRaises(ValueError):
            prepared.feature_mean[0] = 4.0

        leaked = pairs + self._pair(
            criterion="coherence",
            purpose="held_out_evaluation",
            run_id=self._run_for(spec, validation=False),
            suffix="leaked",
            outcome="a",
        )
        audit = CriticCorpusPreparer(
            spec,
            leaked,
            corpus_index_digest="sha256:fixture-corpus",
        ).prepare(require_ready=False).audit

        self.assertFalse(audit["ready"])
        self.assertTrue(audit["run_leakage"])

    def test_training_is_deterministic_and_artifacts_verify(self):
        spec = self._spec()
        digests = []
        scores = []
        with tempfile.TemporaryDirectory(
            prefix="sigillum-pairwise-critic-"
        ) as temporary:
            root = Path(temporary)
            for run in ("one", "two"):
                prepared = CriticCorpusPreparer(
                    spec,
                    self._pairs(spec),
                    corpus_index_digest="sha256:fixture-corpus",
                ).prepare()
                output = root / run
                report = PairwiseCriticTrainer(spec, prepared).train(
                    output_root=output
                )
                verified = verify_critic_artifacts(
                    spec=spec,
                    checkpoint_path=output / "checkpoint.pt",
                    report_path=output / "report.json",
                )
                self.assertEqual("ok", verified["status"])
                digests.append(report["model_digest"])
                scores.append(
                    (
                        report["best_validation_score"],
                        report["held_out_score"],
                    )
                )

        self.assertEqual(digests[0], digests[1])
        self.assertEqual(scores[0], scores[1])

    def test_training_refuses_an_unpinned_corpus(self):
        value = self._spec_dict()
        value["expected_corpus_index_digest"] = None
        spec = CriticLearningSpec.from_dict(value)
        prepared = CriticCorpusPreparer(
            spec,
            self._pairs(spec),
            corpus_index_digest="sha256:fixture-corpus",
        ).prepare()

        with self.assertRaisesRegex(CriticLearningError, "not pinned"):
            PairwiseCriticTrainer(spec, prepared).train(
                output_root=Path("outputs/unused")
            )

    def test_corpus_index_pins_every_evidence_file(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-critic-index-"
        ) as temporary:
            root = Path(temporary)
            source_paths = []
            for filename in (
                "trajectory.jsonl",
                "reviews.jsonl",
                "preferences.jsonl",
            ):
                path = root / "outputs" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
                source_paths.append(f"outputs/{filename}")
            index_path = root / "outputs" / "index.json"
            built = build_critic_corpus_index(
                project_root=root,
                sources=[source_paths],
                output_path=index_path,
            )
            loaded = CriticCorpusIndex.load(index_path)

            self.assertEqual(built["corpus_index_digest"], loaded.digest)
            (root / source_paths[0]).write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(
                CriticLearningError,
                "source digest mismatch",
            ):
                load_critic_pairs(project_root=root, index=loaded)

    def _pairs(self, spec):
        pairs = []
        training_run = self._run_for(spec, validation=False)
        validation_run = self._run_for(spec, validation=True)
        for criterion in CRITERIA:
            scale = "seam" if criterion == "seams" else "global"
            for purpose, run_id, prefix in (
                ("training", training_run, "train"),
                ("training", validation_run, "validation"),
                ("held_out_evaluation", f"held:{criterion}", "held"),
            ):
                pairs.extend(
                    self._pair(
                        criterion=criterion,
                        purpose=purpose,
                        run_id=run_id,
                        suffix=f"{prefix}:a",
                        outcome="a",
                        scale=scale,
                    ),
                )
                pairs.extend(
                    self._pair(
                        criterion=criterion,
                        purpose=purpose,
                        run_id=run_id,
                        suffix=f"{prefix}:b",
                        outcome="b",
                        scale=scale,
                    ),
                )
        return tuple(pairs)

    @staticmethod
    def _pair(
        *,
        criterion,
        purpose,
        run_id,
        suffix,
        outcome,
        scale="global",
    ):
        candidate_id = f"candidate:{criterion}:{suffix}"
        transition_id = f"transition:{criterion}:{suffix}"
        before = {"progress": 0.0, "state": "open"}
        after = {
            "progress": 1.0 if outcome == "a" else -1.0,
            "state": "refined",
        }
        transition = TrajectoryRecord(
            transition_id=transition_id,
            before_snapshot=before,
            before_source="score",
            before_source_digest=(
                "sha256:" + hashlib.sha256(b"score").hexdigest()
            ),
            trajectory_context={
                "run_id": run_id,
                "origin": "deterministic",
                "quality_label": "unrated",
            },
            action={},
            candidates=(
                {
                    "candidate": {
                        "candidate_id": candidate_id,
                        "patch_digest": "sha256:" + "1" * 64,
                        "source_patch": "patch",
                    },
                    "candidate_snapshot": after,
                    "critic_results": (),
                },
            ),
            decision="keep_original",
            after_graph_digest="sha256:" + "2" * 64,
            after_snapshot_digest="sha256:" + "3" * 64,
            raw={},
        )
        variants = (
            {"A": candidate_id, "B": "original"}
            if outcome == "a"
            else {"A": "original", "B": candidate_id}
        )
        review = PairwiseReviewRecord(
            review_id=f"review:{criterion}:{suffix}",
            transition_id=transition_id,
            scale=scale,
            criterion=criterion,
            variants=variants,
            raw={},
        )
        preference = HumanPreferenceRecord(
            preference_id=f"preference:{criterion}:{suffix}",
            review_id=review.review_id,
            transition_id=transition_id,
            outcome=outcome,
            preferred_candidate_id=variants["A" if outcome == "a" else "B"],
            other_candidate_id=variants["B" if outcome == "a" else "A"],
            scale=scale,
            criterion=criterion,
            purpose=purpose,
            raw={},
        )
        return (
            PairwiseExample(
                transition=transition,
                review=review,
                preference=preference,
            ),
        )

    @staticmethod
    def _run_for(spec, *, validation):
        for index in range(1000):
            run_id = f"run:{'validation' if validation else 'training'}:{index}"
            digest = hashlib.sha256(
                f"{spec.validation_salt}\0{run_id}".encode()
            ).digest()
            bucket = int.from_bytes(digest[:8], "big") % spec.validation_modulus
            if (bucket in spec.validation_buckets) == validation:
                return run_id
        raise AssertionError("could not find deterministic split fixture")

    @classmethod
    def _spec(cls):
        return CriticLearningSpec.from_dict(cls._spec_dict())

    @staticmethod
    def _spec_dict():
        return {
            "schema_version": 1,
            "experiment_id": "fixture_pairwise_critics",
            "corpus_index": "outputs/critics/index.json",
            "expected_corpus_index_digest": "sha256:fixture-corpus",
            "output_root": "outputs/critics/model",
            "criteria": [
                {
                    "id": criterion,
                    "scales": ["seam"] if criterion == "seams" else ["global"],
                    "minimum_pairs": {
                        "train": 1,
                        "validation": 1,
                        "held_out_evaluation": 1,
                    },
                    "minimum_runs": {
                        "train": 1,
                        "validation": 1,
                        "held_out_evaluation": 1,
                    },
                }
                for criterion in CRITERIA
            ],
            "validation": {
                "salt": "fixture-validation",
                "modulus": 2,
                "buckets": [0],
            },
            "seed": 7,
            "model": {
                "representation_dim": 8,
                "dropout": 0.0,
            },
            "training": {
                "steps": 6,
                "batch_size_per_criterion": 4,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "validation_interval": 2,
                "patience": 3,
                "gradient_clip": 1.0,
                "evaluation_batch_size": 8,
            },
            "selection_metric": "mean_criterion_balanced_accuracy",
        }


if __name__ == "__main__":
    unittest.main()
