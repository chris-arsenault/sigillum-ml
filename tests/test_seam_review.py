import unittest

import numpy as np

from generation.tools.build_seam_review import (
    _authentic_label,
    review_html,
    synthesize_notes,
)


class SeamReviewTests(unittest.TestCase):
    def test_synthesizer_produces_finite_stereo_pcm(self):
        waveform = synthesize_notes(
            [
                (0.05, 0.4, 60, "Violin", 0),
                (0.2, 0.5, 67, "Horn", 1),
            ],
            total_seconds=1.0,
            sample_rate=8000,
            part_count=2,
        )

        self.assertEqual((8000, 2), waveform.shape)
        self.assertEqual(np.int16, waveform.dtype)
        self.assertTrue(np.any(waveform))

    def test_public_page_contains_review_items_but_no_answer_key(self):
        manifest = {
            "schema_version": 1,
            "review_id": "review:fixture",
            "review_spec_digest": "sha256:" + "1" * 64,
            "prompt": "Which continuation is musically more convincing?",
            "items": [
                {
                    "item_id": "seam-fixture",
                    "position": 1,
                    "audio": {
                        "A": {
                            "file": "audio/A.wav",
                            "digest": "sha256:" + "2" * 64,
                            "duration_seconds": 4.0,
                            "seam_seconds": 2.0,
                        },
                        "B": {
                            "file": "audio/B.wav",
                            "digest": "sha256:" + "3" * 64,
                            "duration_seconds": 4.0,
                            "seam_seconds": 2.0,
                        },
                    },
                }
            ],
        }

        page = review_html(manifest)

        self.assertIn("seam-fixture", page)
        self.assertIn("audio/A.wav", page)
        self.assertNotIn("authentic_label", page)
        self.assertNotIn("model_scores", page)

    def test_blinding_is_deterministic_and_uses_both_labels(self):
        labels = {
            _authentic_label("review-seed", f"item-{index}")
            for index in range(20)
        }

        self.assertEqual({"A", "B"}, labels)
        self.assertEqual(
            _authentic_label("review-seed", "item-4"),
            _authentic_label("review-seed", "item-4"),
        )


if __name__ == "__main__":
    unittest.main()
