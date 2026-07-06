import unittest

from generation.theme_gen import (
    PitchRhythmPin,
    StructuralPin,
    ThemeFrame,
    ThemeKernel,
    generate_theme_batch,
    measure_free_features,
    pins_satisfied,
)


def ql_total(items):
    return sum(float(item[1]) for item in items)


class ThemeGenerationTests(unittest.TestCase):
    def kernel(self):
        return ThemeKernel(
            frame=ThemeFrame(
                bars=4,
                key="C",
                scale=("C", "D", "E", "F", "G", "A", "B"),
                role="heroic",
                lower="C4",
                upper="G5",
                duration_palette=(0.5, 1.0, 1.5, 2.0),
            ),
            pitch_rhythm_pins=(
                PitchRhythmPin(
                    bar=1,
                    beat=1.0,
                    items=(("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("G4", 1.0)),
                    label="opening cell",
                ),
                PitchRhythmPin(
                    bar=4,
                    beat=3.0,
                    items=(("C4", 2.0),),
                    label="cadence landing",
                ),
            ),
            structural_pins=(
                StructuralPin(start_bar=1, end_bar=2, label="question"),
                StructuralPin(start_bar=3, end_bar=4, label="answer"),
            ),
        )

    def corpus(self):
        return (
            (
                ("C4", 1.0),
                ("D4", 1.0),
                ("E4", 1.0),
                ("G4", 1.0),
                ("A4", 0.5),
                ("B4", 0.5),
                ("C5", 1.0),
                ("G4", 2.0),
            ),
            (
                ("G4", 0.5),
                ("E4", 0.5),
                ("D4", 1.0),
                ("F4", 1.5),
                ("E4", 0.5),
                ("C4", 2.0),
            ),
            (
                ("C5", 1.5),
                ("B4", 0.5),
                ("A4", 1.0),
                ("G4", 1.0),
                ("E4", 0.5),
                ("F4", 0.5),
                ("G4", 2.0),
            ),
        )

    def test_generate_theme_batch_preserves_pins_and_selects_spread(self):
        candidates = generate_theme_batch(
            self.kernel(),
            self.corpus(),
            pool_size=40,
            batch_size=6,
            seed=13,
        )

        self.assertEqual(len(candidates), 6)
        unique_lines = {
            tuple((item[0], float(item[1])) for item in candidate.items)
            for candidate in candidates
        }
        self.assertGreater(len(unique_lines), 1)

        for candidate in candidates:
            self.assertAlmostEqual(ql_total(candidate.items), 16.0)
            self.assertEqual(candidate.items[:4], self.kernel().pitch_rhythm_pins[0].items)
            self.assertEqual(candidate.items[-1], ("C4", 2.0))
            self.assertTrue(all(line.endswith("OK") for line in candidate.pin_report))

    def test_free_feature_measurement_subtracts_pinned_regions(self):
        kernel = self.kernel()
        candidate = (
            ("C4", 1.0),
            ("D4", 1.0),
            ("E4", 1.0),
            ("G4", 1.0),
            ("A4", 1.0),
            ("B4", 1.0),
            ("C5", 2.0),
            ("G4", 2.0),
            ("E4", 2.0),
            ("D4", 2.0),
            ("C4", 2.0),
        )

        features = measure_free_features(kernel, candidate)

        self.assertEqual(features.free_bars, (2, 3, 4))
        self.assertTrue(all(line.endswith("OK") for line in pins_satisfied(kernel, candidate)))


if __name__ == "__main__":
    unittest.main()
