import tempfile
import unittest
from pathlib import Path

from generation.partitura_bridge import export_score, single_part_score


class PartituraBridgeTests(unittest.TestCase):
    def test_cross_bar_event_is_materialized_as_tied_noteheads(self):
        score = single_part_score(
            title="Cross-bar bridge",
            items=(("C4", 5.0), ("D4", 1.0)),
            meter="4/4",
            key="C",
            tempo=72,
            beats_per_bar=4.0,
        )

        self.assertIn("C4:4{tie(} | C4:1{tie)} D4:1", score.ruby)

    def test_export_uses_partitura_musicxml_and_midi(self):
        score = single_part_score(
            title="Bridge export",
            items=(("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("F4", 1.0)),
            meter="4/4",
            key="C",
            tempo=72,
            beats_per_bar=4.0,
        )

        with tempfile.TemporaryDirectory() as temp_name:
            xml, midi = export_score(score, Path(temp_name), "bridge")
            self.assertIn("<work-title>Bridge export</work-title>", xml.read_text())
            self.assertEqual(b"MThd", midi.read_bytes()[:4])


if __name__ == "__main__":
    unittest.main()
