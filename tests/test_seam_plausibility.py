import tempfile
import unittest
from pathlib import Path

from generation.tools.run_seam_plausibility import (
    CASES,
    CANDIDATES,
    _base_source,
    _review_manifest,
    _selection_request,
)


class SeamPlausibilityTests(unittest.TestCase):
    def test_all_real_candidates_compile_export_and_return_observations(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-seam-plausibility-"
        ) as temp:
            directory = Path(temp)
            for case in CASES:
                selection, metadata, _paths = _selection_request(
                    directory, case
                )
                observations = selection.to_dict()["candidate_observations"]
                self.assertEqual(len(CANDIDATES), len(selection.candidate_ids))
                self.assertEqual(set(selection.candidate_ids), set(observations))
                self.assertEqual(set(selection.candidate_ids), set(metadata))

    def test_studies_contain_hand_written_anchor_and_open_real_action(self):
        for case in CASES:
            source = _base_source(case)
            self.assertNotIn("# ANCHOR_CONTENT", source)
            self.assertIn("# CANDIDATE_CONTENT", source)
            self.assertIn("pitches <<~PITCHES", source)
            self.assertIn("rhythm <<~RHYTHM", source)
            self.assertIn("span :continuation", source)

    def test_generic_review_manifest_is_blinded_and_links_real_artifacts(self):
        manifest = _review_manifest(
            [
                {
                    "case_id": "fixture",
                    "review": {
                        "review_id": "review:fixture",
                        "title": "Fixture",
                        "audio": {
                            "A": "bundles/item/A.wav",
                            "B": "bundles/item/B.wav",
                        },
                        "midi": {
                            "A": "bundles/item/A.mid",
                            "B": "bundles/item/B.mid",
                        },
                        "musicxml": {
                            "A": "bundles/item/A.musicxml",
                            "B": "bundles/item/B.musicxml",
                        },
                    },
                }
            ]
        )
        serialized = str(manifest)
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["response"]["kind"], "single_choice")
        self.assertEqual(len(manifest["items"][0]["variants"]), 2)
        self.assertIn("MusicXML", serialized)
        self.assertNotIn("coherent_a", serialized)
        self.assertNotIn("subtle_wrong", serialized)


if __name__ == "__main__":
    unittest.main()
