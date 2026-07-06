import unittest

from generation.theme_gen import frame, harm, kernel, phrase, pin
from generation.theme_gen.kerneldsl import (
    HarmonicPin,
    PitchRhythmPin,
    StructuralPin,
    ThemeFrame,
    ThemeKernel,
)


class KernelDslTests(unittest.TestCase):
    def test_dsl_kernel_equals_dataclass_kernel(self):
        dsl = kernel(
            frame(4, key="C", role="heroic", lower="C4", upper="G5", durations=(0.5, 1.0, 1.5, 2.0)),
            pins=[pin(1, 1.0, ("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0), label="head")],
            harmony=[harm(1, "I", "C", "E", "G")],
            structure=[phrase(1, 4, "period", cadence="authentic cadence")],
        )
        direct = ThemeKernel(
            frame=ThemeFrame(
                bars=4, meter="4/4", beats_per_bar=4.0, key="C",
                scale=("C", "D", "E", "F", "G", "A", "B"), role="heroic",
                lower="C4", upper="G5", duration_palette=(0.5, 1.0, 1.5, 2.0),
            ),
            pitch_rhythm_pins=(
                PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0)), label="head"),
            ),
            harmonic_pins=(HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),),
            structural_pins=(StructuralPin(start_bar=1, end_bar=4, label="period", cadence="authentic cadence"),),
        )
        self.assertEqual(dsl, direct)


if __name__ == "__main__":
    unittest.main()
