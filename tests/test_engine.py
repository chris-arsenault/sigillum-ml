import unittest
from random import Random

from music21 import pitch as m21pitch

from generation.theme_gen.engine import (
    MAX_BACKTRACKS,
    MAX_LANDING_LEAP,
    _free_segments,
    _generate_constrained_items,
    _generate_segment,
    _legal_pitches,
    _palette_ticks,
    _search_segment,
    _segment_feasible,
)
from generation.theme_gen.engine import generate_theme_batch, pin_outcome, pins_satisfied


def _harmonic_verdict(report, bar):
    line = next(l for l in report if l.startswith(f"harmonic b{bar} "))
    return line.endswith("OK")


def _structural_verdict(report, start_bar, end_bar):
    line = next(l for l in report if l.startswith(f"structural b{start_bar}-{end_bar} "))
    return line.endswith("OK")
from generation.theme_gen.kerneldsl import (
    HarmonicPin,
    PitchRhythmPin,
    StructuralPin,
    ThemeFrame,
    ThemeKernel,
    _cadence_target_pc,
    _diatonic_triads,
    _dominant_pc,
    _duration_palette,
    _frame_bounds,
    _harmonic_pcs_for_offset,
    _scale_degree,
    _scale_midis,
    _tonic_pc,
)
from generation.theme_gen._common import _duration_to_ticks
from generation.theme_gen.corpus import default_theme_corpus
from generation.theme_gen.model import MarkovMelodyModel


def make_frame(**kw):
    base = dict(bars=4, key="C", scale=("C", "D", "E", "F", "G", "A", "B"),
                lower="C4", upper="C6", duration_palette=(0.5, 1.0, 1.5, 2.0))
    base.update(kw)
    return ThemeFrame(**base)


class HarmonyGeometryTests(unittest.TestCase):
    def test_key_and_cadence_geometry(self):
        frame = make_frame()  # C major
        self.assertEqual(_tonic_pc(frame), 0)
        self.assertEqual(_dominant_pc(frame), 7)
        self.assertEqual(_scale_degree(frame, 7), 5)
        self.assertIsNone(_scale_degree(frame, 1))  # C#, chromatic
        self.assertEqual(_diatonic_triads(frame)[1], frozenset({0, 4, 7}))
        self.assertEqual(_diatonic_triads(frame)[5], frozenset({7, 11, 2}))
        self.assertEqual(_cadence_target_pc(frame, "authentic cadence"), 0)
        self.assertEqual(_cadence_target_pc(frame, "half cadence"), 7)
        self.assertIsNone(_cadence_target_pc(frame, ""))


class HarmonicImplicationTests(unittest.TestCase):
    def kernel(self):
        return ThemeKernel(
            frame=make_frame(),
            harmonic_pins=(HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),),
        )

    def filler(self):
        # bars 2-4: any scale filler so the full candidate is 16 ql (only bar 1 is checked)
        return [("G4", 1.0)] * 4 + [("F4", 1.0)] * 4 + [("C4", 1.0)] * 4

    def test_implies_chord_when_downbeat_chord_tone_and_no_competing_triad(self):
        items = [("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("E4", 1.0)] + self.filler()
        self.assertTrue(_harmonic_verdict(pins_satisfied(self.kernel(), items), 1))

    def test_rejects_non_chord_tone_on_downbeat(self):
        items = [("D4", 1.0), ("E4", 1.0), ("G4", 1.0), ("E4", 1.0)] + self.filler()
        self.assertFalse(_harmonic_verdict(pins_satisfied(self.kernel(), items), 1))

    def test_rejects_bar_implying_a_competing_triad(self):
        # downbeat C is a chord tone of I, but C-A-F-A best-matches IV (F-A-C), not I.
        items = [("C4", 1.0), ("A4", 1.0), ("F4", 1.0), ("A4", 1.0)] + self.filler()
        self.assertFalse(_harmonic_verdict(pins_satisfied(self.kernel(), items), 1))


class CadenceArticulationTests(unittest.TestCase):
    def kernel(self, cadence):
        return ThemeKernel(
            frame=make_frame(bars=2),
            structural_pins=(StructuralPin(start_bar=1, end_bar=2, label="phrase", cadence=cadence),),
        )

    def ending_on(self, last):
        # 2 bars = 8 ql: 7 ql of filler then a 1 ql final note
        return [("E4", 1.0)] * 7 + [(last, 1.0)]

    def test_authentic_requires_tonic_final(self):
        ok = pins_satisfied(self.kernel("authentic cadence"), self.ending_on("C5"))
        self.assertTrue(_structural_verdict(ok, 1, 2))
        bad = pins_satisfied(self.kernel("authentic cadence"), self.ending_on("D5"))
        self.assertFalse(_structural_verdict(bad, 1, 2))

    def test_half_requires_dominant_final(self):
        ok = pins_satisfied(self.kernel("half cadence"), self.ending_on("G4"))
        self.assertTrue(_structural_verdict(ok, 1, 2))
        bad = pins_satisfied(self.kernel("half cadence"), self.ending_on("C5"))
        self.assertFalse(_structural_verdict(bad, 1, 2))

    def test_unrecognized_cadence_is_metadata_ok(self):
        report = pins_satisfied(self.kernel(""), self.ending_on("D5"))
        self.assertTrue(_structural_verdict(report, 1, 2))


class LegalMoveTests(unittest.TestCase):
    def test_legal_pitches_are_the_full_chromatic_range(self):
        # No scale or chord-tone filtering: the model chooses among every chromatic pitch in
        # range (so non-scale corpus content is reachable). Range is the only hard pitch bound.
        kernel = ThemeKernel(
            frame=make_frame(),
            harmonic_pins=(HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),),
        )
        low, high = _frame_bounds(kernel.frame)
        self.assertEqual(_legal_pitches(kernel, 0.0), list(range(low, high + 1)))
        self.assertEqual(_legal_pitches(kernel, 1.5), list(range(low, high + 1)))
        # A non-scale pitch (C#) is now legal, where the old scale filter discarded it.
        self.assertIn(low + 1, _legal_pitches(kernel))

    def test_palette_ticks(self):
        frame = make_frame()
        expected = tuple(sorted({_duration_to_ticks(d) for d in _duration_palette(frame)}))
        self.assertEqual(_palette_ticks(frame), expected)


class FeasibilityTests(unittest.TestCase):
    def kernel(self):
        return ThemeKernel(frame=make_frame())

    def test_fillable_span_with_reachable_pin_is_feasible(self):
        # one bar (48 ticks) toward a pin pitch equal to a scale pitch in range
        self.assertTrue(_segment_feasible(self.kernel(), 0.0, 48, 60, 60))

    def test_unfillable_rhythm_is_infeasible(self):
        # 7 ticks cannot be composed from the palette {6,12,18,24}
        self.assertFalse(_segment_feasible(self.kernel(), 0.0, 7, 60, 60))

    def test_landing_leap_bounds_feasibility(self):
        # empty span: feasibility reduces to the landing leap from prev_pitch
        self.assertTrue(_segment_feasible(self.kernel(), 4.0, 0, 60, 60 + MAX_LANDING_LEAP))
        self.assertFalse(_segment_feasible(self.kernel(), 4.0, 0, 60, 60 + MAX_LANDING_LEAP + 1))


class SegmentSamplerTests(unittest.TestCase):
    def kernel(self):
        return ThemeKernel(
            frame=make_frame(),
            harmonic_pins=(HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),),
        )

    def model(self):
        return MarkovMelodyModel.from_corpus(default_theme_corpus())

    def test_unreachable_pin_is_abandoned(self):
        # Pin pitch is far outside the frame range: no scale pitch lands within the leap,
        # so the feasibility-gated sampler must abandon (None) rather than mis-land.
        result = _generate_segment(self.kernel(), self.model(), Random(1), 0.0, 4.0, 60, 0, None, 100)
        self.assertIsNone(result)

    def test_feasible_segment_satisfies_contract_and_is_deterministic(self):
        kernel, model = self.kernel(), self.model()
        result = _generate_segment(kernel, model, Random(7), 0.0, 4.0, None, 0, None, 67)
        self.assertIsNotNone(result)
        items, *_ = result
        self.assertAlmostEqual(sum(float(it[1]) for it in items), 4.0)  # exact fill
        offset = 0.0
        for name, ql in items:
            midi = int(m21pitch.Pitch(name).midi)
            self.assertIn(midi, _legal_pitches(kernel, offset))  # every note legal at offset
            offset += ql
        last = int(m21pitch.Pitch(items[-1][0]).midi)
        self.assertLessEqual(abs(67 - last), MAX_LANDING_LEAP)  # lands cleanly
        again, *_ = _generate_segment(kernel, model, Random(7), 0.0, 4.0, None, 0, None, 67)
        self.assertEqual(items, again)  # deterministic under seed


class BacktrackTests(unittest.TestCase):
    def kernel(self):
        return ThemeKernel(
            frame=make_frame(),
            harmonic_pins=(HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),),
        )

    def model(self):
        return MarkovMelodyModel.from_corpus(default_theme_corpus())

    def test_over_tight_segment_abandons_within_budget(self):
        # Unreachable pin: no feasible move; the search abandons (None) without hanging,
        # staying within the backtrack budget.
        result, backtracks = _search_segment(
            self.kernel(), self.model(), Random(1), 0.0, 4.0, 60, 0, None, 100
        )
        self.assertIsNone(result)
        self.assertLessEqual(backtracks, MAX_BACKTRACKS)

    def test_normal_segment_uses_no_backtracks(self):
        result, backtracks = _search_segment(
            self.kernel(), self.model(), Random(7), 0.0, 4.0, None, 0, None, 67
        )
        self.assertIsNotNone(result)
        self.assertEqual(backtracks, 0)


class FreeSegmentTests(unittest.TestCase):
    def test_free_segments_for_head_pinned_cadence_kernel(self):
        # head pin fills bar 1; a frame-final authentic cadence lands the final note.
        kernel = ThemeKernel(
            frame=make_frame(bars=2, lower="C4", upper="C6"),
            pitch_rhythm_pins=(
                PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0)), label="head"),
            ),
            structural_pins=(StructuralPin(start_bar=1, end_bar=2, label="phrase", cadence="authentic cadence"),),
        )
        segments = _free_segments(kernel)
        # one gap: from the end of the bar-1 head (offset 4.0) to the cadence end (8.0),
        # landing on the tonic pitch class (0), no pin pitch.
        self.assertEqual(segments, [(4.0, 8.0, None, 0)])


class IntegrationTests(unittest.TestCase):
    def tight_kernel(self):
        # Harmonic pin on every bar + pitch/rhythm pins: a tightly-pinned, satisfiable kernel.
        return ThemeKernel(
            frame=make_frame(role="heroic", lower="C4", upper="G5"),
            pitch_rhythm_pins=(
                PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0)), label="head"),
                PitchRhythmPin(bar=4, beat=3.0, items=(("C4", 2.0),), label="cadence"),
            ),
            harmonic_pins=(
                HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),
                HarmonicPin(bar=2, label="V", chord_tones=("G", "B", "D")),
                HarmonicPin(bar=3, label="IV", chord_tones=("F", "A", "C")),
                HarmonicPin(bar=4, label="I", chord_tones=("C", "E", "G")),
            ),
        )

    def test_tightly_pinned_kernel_yields_full_hard_pin_batch(self):
        # Hard pins (pitch/rhythm — the head and the cadence note) are satisfied on every kept
        # candidate; harmonic-implication pins are now soft (guided, scored, not a drop gate).
        kernel = self.tight_kernel()
        batch = generate_theme_batch(kernel, pool_size=40, batch_size=6, seed=5)
        self.assertEqual(len(batch), 6)
        for candidate in batch:
            self.assertTrue(pin_outcome(kernel, candidate.items)[0])

    def test_impossible_segment_abandons_to_clean_error(self):
        # The span before the C6 pin cannot land within MAX_LANDING_LEAP given the G4 ceiling,
        # so every attempt abandons; the batch surfaces a clean RuntimeError, not a crash.
        impossible = ThemeKernel(
            frame=ThemeFrame(bars=2, lower="C4", upper="G4", duration_palette=(0.5, 1.0, 1.5, 2.0)),
            pitch_rhythm_pins=(
                PitchRhythmPin(bar=1, beat=1.0, items=(("C4", 1.0),), label="a"),
                PitchRhythmPin(bar=2, beat=1.0, items=(("C6", 1.0),), label="b"),
            ),
        )
        with self.assertRaises(RuntimeError):
            generate_theme_batch(impossible, pool_size=10, batch_size=3, seed=1)


class GenerationEnforcementTests(unittest.TestCase):
    def cadence_kernel(self):
        chords = ("C", "E", "G")
        return ThemeKernel(
            frame=make_frame(bars=2, lower="C4", upper="C6"),
            harmonic_pins=(
                HarmonicPin(bar=1, label="I", chord_tones=chords),
                HarmonicPin(bar=2, label="I", chord_tones=chords),
            ),
            structural_pins=(StructuralPin(start_bar=1, end_bar=2, label="phrase", cadence="authentic cadence"),),
        )

    def two_cadence_kernel(self):
        return ThemeKernel(
            frame=make_frame(bars=4, lower="C4", upper="C6"),
            harmonic_pins=(
                HarmonicPin(bar=2, label="V", chord_tones=("G", "B", "D")),
                HarmonicPin(bar=4, label="I", chord_tones=("C", "E", "G")),
            ),
            structural_pins=(
                StructuralPin(start_bar=1, end_bar=2, label="antecedent", cadence="half cadence"),
                StructuralPin(start_bar=3, end_bar=4, label="consequent", cadence="authentic cadence"),
            ),
        )

    def test_raw_generation_lands_mid_frame_and_final_cadences(self):
        # The mid-frame antecedent half cadence (unpinned) and the final authentic cadence
        # must both be landed by generation itself, not only by rejection.
        kernel = self.two_cadence_kernel()
        model = MarkovMelodyModel.from_corpus(default_theme_corpus())
        dominant, tonic = _dominant_pc(kernel.frame), _tonic_pc(kernel.frame)
        for seed in range(5):
            items = _generate_constrained_items(kernel, model, Random(seed))
            self.assertIsNotNone(items)
            offset, antecedent_final = 0.0, None
            for item in items:
                if item[0] is not None and abs(offset + float(item[1]) - 8.0) < 1e-6:
                    antecedent_final = item[0]  # note ending at the bar-2 boundary
                offset += float(item[1])
            self.assertIsNotNone(antecedent_final)
            self.assertEqual(int(m21pitch.Pitch(antecedent_final).midi) % 12, dominant)
            last = next(item for item in reversed(items) if item[0] is not None)
            self.assertEqual(int(m21pitch.Pitch(last[0]).midi) % 12, tonic)

    def test_raw_generation_lands_the_frame_final_cadence(self):
        # The RAW generator (pre-filter) must land the cadence target itself, so cadence
        # kernels fill without relying on rejection. Without cadence_pc threading the raw
        # final note is rarely the tonic.
        kernel = self.cadence_kernel()
        model = MarkovMelodyModel.from_corpus(default_theme_corpus())
        tonic = _tonic_pc(kernel.frame)
        for seed in range(8):
            items = _generate_constrained_items(kernel, model, Random(seed))
            self.assertIsNotNone(items)
            last = next(item for item in reversed(items) if item[0] is not None)
            self.assertEqual(int(m21pitch.Pitch(last[0]).midi) % 12, tonic)


class M4ExitTests(unittest.TestCase):
    HEAD = [("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("C5", 1.0)]

    def kernel(self):
        return ThemeKernel(
            frame=make_frame(bars=4, lower="C4", upper="C6"),
            pitch_rhythm_pins=(PitchRhythmPin(bar=1, beat=1.0, items=tuple(self.HEAD), label="head"),),
            harmonic_pins=(
                HarmonicPin(bar=1, label="I", chord_tones=("C", "E", "G")),
                HarmonicPin(bar=2, label="V", chord_tones=("G", "B", "D")),
                HarmonicPin(bar=3, label="V", chord_tones=("G", "B", "D")),
                HarmonicPin(bar=4, label="I", chord_tones=("C", "E", "G")),
            ),
            structural_pins=(StructuralPin(start_bar=1, end_bar=4, label="period", cadence="authentic cadence"),),
        )

    def test_cadence_kernel_yields_full_articulating_batch(self):
        # The cadence is still forced at generation (cadence_pc), so every candidate articulates
        # it (structural pins OK) and the hard head pin holds — even though harmonic-implication
        # pins are now soft and a candidate may miss them.
        kernel = self.kernel()
        batch = generate_theme_batch(kernel, pool_size=30, batch_size=6, seed=11)
        self.assertEqual(len(batch), 6)
        for candidate in batch:
            self.assertTrue(pin_outcome(kernel, candidate.items)[0])
            structural = [line for line in candidate.pin_report if line.startswith("structural ")]
            self.assertTrue(structural and all(line.endswith("OK") for line in structural))

    def test_candidate_that_old_checks_accepted_is_now_rejected(self):
        # Final bar contains chord tones (the old ">=1 chord tone" rule passed it) and the
        # old structural check was unconditional OK, yet this candidate ends on D (not the
        # tonic): the strengthened cadence check must reject it.
        barV = [("G4", 1.0), ("B4", 1.0), ("D5", 1.0), ("B4", 1.0)]
        bar4 = [("C5", 1.0), ("E5", 1.0), ("G5", 1.0), ("D5", 1.0)]  # chord tones, ends on D
        report = pins_satisfied(self.kernel(), self.HEAD + barV + barV + bar4)
        self.assertFalse(all(line.endswith("OK") for line in report))
        self.assertFalse(_structural_verdict(report, 1, 4))  # the cadence is what fails


if __name__ == "__main__":
    unittest.main()
