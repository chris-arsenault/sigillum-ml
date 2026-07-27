import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from generation.composition import (
    ORIGINAL_CANDIDATE_ID,
    CandidateProposal,
    CompositionDataset,
    LearnedCompositionProvider,
    LearnedCritic,
    LearnedCriticResult,
    LearnedPolicy,
    LearnedProposer,
    ProposalRequest,
    ProposalResponse,
    ProtocolError,
    SelectionRequest,
    SelectionResponse,
)
from generation.composition.evaluation_store import trajectory_effort

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "composition_kernel_study.rb"
PARTITURA = ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"

VALID_BASS_PATCH = """\
diff --git a/composition_kernel_study.rb b/composition_kernel_study.rb
--- a/composition_kernel_study.rb
+++ b/composition_kernel_study.rb
@@ -32,8 +32,8 @@
                 part: :flute, role: :foreground, at: "bar 1 beat 1"

       phrase :statement_bass, surface: :absolute do
-        pitch_bars "C3 G2"
-        rhythm_bars "2 2"
+        pitch_bars "C3 G2 | B2 G2"
+        rhythm_bars "2 2 | 2 2"
       end
       placement :statement_bass, id: :statement_bass_cello,
                 part: :cello, role: :bass_line, at: "bar 1 beat 1"
"""

INVALID_BASS_PATCH = """\
diff --git a/composition_kernel_study.rb b/composition_kernel_study.rb
--- a/composition_kernel_study.rb
+++ b/composition_kernel_study.rb
@@ -32,8 +32,8 @@
                 part: :flute, role: :foreground, at: "bar 1 beat 1"

       phrase :statement_bass, surface: :absolute do
-        pitch_bars "C3 G2"
-        rhythm_bars "2 2"
+        pitch_bars "C3 G2 | B2 G2
+        rhythm_bars "2 2 | 2 2"
       end
       placement :statement_bass, id: :statement_bass_cello,
                 part: :cello, role: :bass_line, at: "bar 1 beat 1"
"""


def partitura(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("ruby", str(PARTITURA), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


class _Provider:
    def propose(self, request: ProposalRequest) -> ProposalResponse:
        return ProposalResponse.create(
            request,
            producer="test-proposer",
            candidates=(),
        )

    def select(self, request: SelectionRequest) -> SelectionResponse:
        return SelectionResponse.create(
            request,
            producer="test-policy",
            selected_candidate_id=ORIGINAL_CANDIDATE_ID,
            reason="Keep the explicit original.",
        )


class _Critic:
    def evaluate(self, request: SelectionRequest) -> tuple[LearnedCriticResult, ...]:
        del request
        return ()


class _Policy:
    def select(
        self,
        request: SelectionRequest,
        critic_results: tuple[LearnedCriticResult, ...],
    ) -> SelectionResponse:
        del critic_results
        return SelectionResponse.create(
            request,
            producer="test-policy",
            selected_candidate_id=ORIGINAL_CANDIDATE_ID,
            reason="Keep the explicit original.",
        )


class CompositionProtocolTests(unittest.TestCase):
    def test_protocol_types_are_the_ml_extension_points(self):
        provider = _Provider()

        self.assertIsInstance(provider, LearnedProposer)
        self.assertIsInstance(provider, LearnedCompositionProvider)
        self.assertIsInstance(_Critic(), LearnedCritic)
        self.assertIsInstance(_Policy(), LearnedPolicy)

    def test_ruby_and_python_exchange_proposals_evidence_and_selection(self):
        with tempfile.TemporaryDirectory(prefix="sigillum-ml-protocol-") as temp:
            directory = Path(temp)
            source = directory / FIXTURE.name
            trajectory = directory / "trajectory.jsonl"
            proposals = directory / "proposals.json"
            selection = directory / "selection.json"
            reviews = directory / "reviews.jsonl"
            preferences = directory / "preferences.jsonl"
            review_bundles = directory / "review-bundles"
            shutil.copyfile(FIXTURE, source)

            observed = partitura(
                "observe",
                str(source),
                "--trajectory",
                str(trajectory),
                "--no-export",
            )
            self.assertEqual(observed.returncode, 0, observed.stderr)
            request = ProposalRequest.from_json(observed.stdout)
            self.assertEqual(
                request.action["base_snapshot_digest"],
                request.snapshot["snapshot_digest"],
            )
            with self.assertRaises(TypeError):
                request.action["lens"] = "not-mutable"

            candidate = CandidateProposal.inline(
                request,
                source_patch=VALID_BASS_PATCH,
                description="Complete the scheduled bass through the statement.",
            )
            rejected_candidate = CandidateProposal.inline(
                request,
                source_patch=INVALID_BASS_PATCH,
                description="A syntactically invalid alternative retained as evidence.",
            )
            proposal = ProposalResponse.create(
                request,
                producer="test-learned-proposer",
                candidates=(candidate, rejected_candidate),
            )
            proposals.write_text(proposal.to_json(), encoding="utf-8")

            evaluated = partitura(
                "evaluate",
                str(source),
                "--trajectory",
                str(trajectory),
                "--proposals",
                str(proposals),
                "--no-export",
            )
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            selection_request = SelectionRequest.from_json(evaluated.stdout)
            self.assertEqual(
                selection_request.candidate_ids,
                (candidate.candidate_id, rejected_candidate.candidate_id),
            )
            mechanical = selection_request.assessments[0]["critic_results"][0]
            self.assertEqual(mechanical["scale"], "mechanical")
            self.assertTrue(mechanical["passed"])
            self.assertNotIn(
                "source_patch", selection_request.assessments[0]["candidate"]
            )

            learned = LearnedCriticResult(
                critic="test-learned-local",
                scale="local",
                target_path=candidate.target_path,
                candidate_id=candidate.candidate_id,
                features={"learned_balance": 0.6},
                score=0.8,
                confidence=0.7,
            )
            response = SelectionResponse.create(
                selection_request,
                producer="test-learned-policy",
                selected_candidate_id=candidate.candidate_id,
                reason="Prefer the mechanically valid completion.",
                critic_results=(learned,),
            )
            selection.write_text(response.to_json(), encoding="utf-8")

            stepped = partitura(
                "step",
                str(source),
                "--trajectory",
                str(trajectory),
                "--proposals",
                str(proposals),
                "--selection",
                str(selection),
                "--no-export",
            )
            self.assertEqual(stepped.returncode, 0, stepped.stderr)
            result = json.loads(stepped.stdout)
            self.assertEqual(result["transition"]["decision"], "accept")
            self.assertIn(
                'pitch_bars "C3 G2 | B2 G2"',
                source.read_text(encoding="utf-8"),
            )
            transition = json.loads(trajectory.read_text(encoding="utf-8"))
            self.assertEqual(transition["selection"]["producer"], "test-learned-policy")
            self.assertEqual(
                transition["candidates"][0]["critic_results"][1]["critic"],
                "test-learned-local",
            )
            self.assertEqual(transition["schema_version"], 2)
            self.assertEqual(transition["before_source_digest"], request.source_digest)
            self.assertEqual(
                transition["before_snapshot"]["snapshot_digest"],
                request.snapshot["snapshot_digest"],
            )
            self.assertEqual(
                transition["candidates"][1]["candidate"]["source_patch"],
                INVALID_BASS_PATCH,
            )
            effort = trajectory_effort(
                trajectory, model_call_count=2, wall_seconds=1.25
            )
            self.assertEqual(effort.candidate_count, 2)
            self.assertEqual(effort.mechanically_valid_candidate_count, 1)
            self.assertEqual(effort.accepted_edit_count, 1)

            training_review = self._create_review(
                trajectory=trajectory,
                reviews=reviews,
                output=review_bundles,
                transition_id=transition["transition_id"],
                candidate_id=candidate.candidate_id,
                seed="training-review",
            )
            held_out_review = self._create_review(
                trajectory=trajectory,
                reviews=reviews,
                output=review_bundles,
                transition_id=transition["transition_id"],
                candidate_id=candidate.candidate_id,
                seed="held-out-review",
            )
            self._record_preference(
                reviews,
                preferences,
                training_review["review_id"],
                "training",
            )
            self._record_preference(
                reviews,
                preferences,
                held_out_review["review_id"],
                "held_out_evaluation",
            )

            dataset = CompositionDataset.from_jsonl(trajectory, reviews, preferences)
            self.assertEqual(dataset.trajectories[0].origin, "deterministic")
            self.assertEqual(dataset.trajectories[0].quality_label, "unrated")
            self.assertEqual(
                set(dataset.trajectories[0].candidate_ids),
                {candidate.candidate_id, rejected_candidate.candidate_id},
            )
            self.assertEqual(len(dataset.training_pairs()), 1)
            self.assertEqual(len(dataset.held_out_pairs()), 1)
            self.assertNotEqual(
                dataset.training_pairs()[0].review.review_id,
                dataset.held_out_pairs()[0].review.review_id,
            )

    def test_protocol_rejects_python_claims_of_mechanical_authority(self):
        with self.assertRaises(ProtocolError):
            LearnedCriticResult(
                critic="incorrect-boundary",
                scale="mechanical",
                target_path="span:any",
                candidate_id="candidate:any",
                passed=True,
            )

    def test_response_must_select_a_candidate_offered_by_ruby(self):
        selection_request = SelectionRequest(
            request_id="selection:request",
            proposal_request_id="proposal:request",
            snapshot={"snapshot_digest": "opaque"},
            action={"action_id": "action:opaque"},
            original_candidate_id=ORIGINAL_CANDIDATE_ID,
            assessments=(
                {
                    "candidate": {"candidate_id": "candidate:known"},
                    "critic_results": (),
                },
            ),
        )

        with self.assertRaises(ProtocolError):
            SelectionResponse.create(
                selection_request,
                producer="test-policy",
                selected_candidate_id="candidate:unknown",
                reason="This candidate was not offered.",
            )

    def _create_review(
        self,
        *,
        trajectory: Path,
        reviews: Path,
        output: Path,
        transition_id: str,
        candidate_id: str,
        seed: str,
    ) -> dict[str, object]:
        reviewed = partitura(
            "review",
            "--trajectory",
            str(trajectory),
            "--reviews",
            str(reviews),
            "--output",
            str(output),
            "--transition",
            transition_id,
            "--candidate",
            candidate_id,
            "--against",
            ORIGINAL_CANDIDATE_ID,
            "--scale",
            "global",
            "--seed",
            seed,
        )
        self.assertEqual(reviewed.returncode, 0, reviewed.stderr)
        payload = json.loads(reviewed.stdout)
        public_manifest = json.loads(
            (Path(payload["bundle"]) / "review.json").read_text(encoding="utf-8")
        )
        self.assertNotIn(candidate_id, json.dumps(public_manifest))
        self.assertEqual(
            {item["label"] for item in public_manifest["variants"]}, {"A", "B"}
        )
        return payload

    def _record_preference(
        self,
        reviews: Path,
        preferences: Path,
        review_id: object,
        purpose: str,
    ) -> None:
        preferred = partitura(
            "preference",
            "--reviews",
            str(reviews),
            "--preferences",
            str(preferences),
            "--review",
            str(review_id),
            "--outcome",
            "a",
            "--rater",
            "rater:protocol-test",
            "--purpose",
            purpose,
            "--reason",
            "A has the clearer large-scale trajectory.",
            "--confidence",
            "0.75",
        )
        self.assertEqual(preferred.returncode, 0, preferred.stderr)


if __name__ == "__main__":
    unittest.main()
