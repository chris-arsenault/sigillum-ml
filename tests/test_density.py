import unittest

from generation.theme_gen.density import kernel_density
from generation.theme_gen.report import render_candidate_report, render_kernel_density
from generation.theme_gen.kerneldsl import (
    HarmonicPin,
    PitchRhythmPin,
    StructuralPin,
    ThemeFrame,
    ThemeKernel,
)


def make_frame(**kw):
    base = dict(bars=4, key="C", scale=("C", "D", "E", "F", "G", "A", "B"),
                lower="C4", upper="C6", duration_palette=(0.5, 1.0, 1.5, 2.0))
    base.update(kw)
    return ThemeFrame(**base)


def fully_pinned_kernel():
    # one bar entirely covered by a pitch/rhythm pin
    return ThemeKernel(
        frame=make_frame(bars=1),
        pitch_rhythm_pins=(
            PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0)), label="all"),
        ),
    )


def open_kernel():
    return ThemeKernel(frame=make_frame(bars=4))


class DensityTests(unittest.TestCase):
    def test_fully_pinned_has_near_zero_free_space(self):
        d = kernel_density(fully_pinned_kernel())
        self.assertAlmostEqual(d.free_space_fraction, 0.0)
        self.assertAlmostEqual(d.density, 1.0)
        self.assertAlmostEqual(d.pinned_ql + d.free_ql, d.total_ql)

    def test_open_kernel_is_all_free_space(self):
        d = kernel_density(open_kernel())
        self.assertAlmostEqual(d.free_space_fraction, 1.0)
        self.assertAlmostEqual(d.pinned_ql, 0.0)

    def test_fully_pinned_has_one_realization_and_zero_spread(self):
        d = kernel_density(fully_pinned_kernel())
        self.assertEqual(d.feasible_realizations, 1)
        self.assertFalse(d.realizations_saturated)
        self.assertAlmostEqual(d.expected_spread_bits, 0.0)

    def test_open_kernel_saturates_with_wide_spread(self):
        d = kernel_density(open_kernel())
        self.assertTrue(d.realizations_saturated)
        self.assertGreater(d.expected_spread_bits, 10.0)
        # deterministic
        self.assertEqual(d.feasible_realizations, kernel_density(open_kernel()).feasible_realizations)

    def test_pin_counts(self):
        kernel = ThemeKernel(
            frame=make_frame(bars=2),
            pitch_rhythm_pins=(PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0),)),),
            harmonic_pins=(HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),),
            structural_pins=(StructuralPin(start_bar=1, end_bar=2, label="p", cadence="authentic cadence"),),
        )
        d = kernel_density(kernel)
        self.assertEqual((d.n_pitch_pins, d.n_harmonic_pins, d.n_structural_pins), (1, 1, 1))


class M5ExitTests(unittest.TestCase):
    def near_full_kernel(self):
        # one bar, 3.5 ql pinned, leaving a single 0.5 ql free note.
        return ThemeKernel(
            frame=make_frame(bars=1),
            pitch_rhythm_pins=(
                PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 0.5)), label="most"),
            ),
        )

    def test_tight_kernel_narrow_open_kernel_wide(self):
        tight = kernel_density(self.near_full_kernel())
        wide = kernel_density(open_kernel())
        self.assertLess(tight.free_space_fraction, 0.2)  # near-zero free space
        self.assertLess(tight.expected_spread_bits, 6.0)  # narrow spread
        self.assertGreater(wide.free_space_fraction, tight.free_space_fraction)
        self.assertGreater(wide.expected_spread_bits, tight.expected_spread_bits + 5.0)

    def test_tight_kernel_report_warns_of_near_duplicates(self):
        density = kernel_density(self.near_full_kernel())
        block = render_kernel_density(self.near_full_kernel(), batch_size=density.feasible_realizations + 5)
        self.assertIn("near-duplicates", block)


class ReportTests(unittest.TestCase):
    def test_render_kernel_density_block(self):
        block = render_kernel_density(fully_pinned_kernel())
        self.assertIn("free space 0%", block)
        self.assertIn("Expected spread: 0.0 bits", block)

    def test_candidate_report_includes_density(self):
        report = render_candidate_report([], open_kernel())
        self.assertIn("free space", report)
        self.assertIn("Expected spread", report)


if __name__ == "__main__":
    unittest.main()
