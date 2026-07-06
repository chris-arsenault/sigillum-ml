import unittest

from generation.theme_gen._common import _TimedItem
from random import Random

from generation.theme_gen.features import (
    _bar_signature,
    _feature_vector,
    _roll_contour,
    _role_contour_jitter,
    _signature_similarity,
    measure_free_features,
)
from generation.theme_gen.kerneldsl import ThemeFrame, ThemeKernel


def sig(items):
    return _bar_signature([_TimedItem(0.0, it) for it in items])


class SimilarityTransformTests(unittest.TestCase):
    A = [("C4", 0.5), ("D4", 0.5), ("E4", 0.5), ("F4", 0.5)]      # intervals +2,+2,+2
    TRANSPOSED = [("G4", 0.5), ("A4", 0.5), ("B4", 0.5), ("C5", 0.5)]
    INVERSION = [("C4", 0.5), ("B-3", 0.5), ("A-3", 0.5), ("G-3", 0.5)]  # -2,-2,-2
    AUGMENTATION = [("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("F4", 1.0)]  # +2,+2,+2 rhythm x2

    def test_transposition_still_detected(self):
        self.assertGreaterEqual(_signature_similarity(sig(self.A), sig(self.TRANSPOSED)), 0.95)

    def test_inversion_detected(self):
        self.assertGreaterEqual(_signature_similarity(sig(self.A), sig(self.INVERSION)), 0.85)

    def test_augmentation_detected(self):
        self.assertGreaterEqual(_signature_similarity(sig(self.A), sig(self.AUGMENTATION)), 0.85)


def open_kernel(bars=2):
    return ThemeKernel(frame=ThemeFrame(bars=bars, key="C", scale=("C", "D", "E", "F", "G", "A", "B"),
                                        lower="C4", upper="C6", duration_palette=(0.5, 1.0, 1.5, 2.0)))


class ContourDetailTests(unittest.TestCase):
    def test_detail_distinguishes_same_mean_shapes(self):
        kernel = open_kernel(bars=2)
        filler = [("C4", 1.0), ("C4", 1.0), ("C4", 1.0), ("C4", 1.0)]  # bar 2, shared
        rising = [("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("E4", 1.0)] + filler
        falling = [("E4", 1.0), ("G4", 1.0), ("E4", 1.0), ("C4", 1.0)] + filler  # same multiset, reversed shape
        fr = measure_free_features(kernel, rising)
        ff = measure_free_features(kernel, falling)
        self.assertEqual(fr.contour, ff.contour)  # per-bar means identical
        self.assertNotEqual(fr.contour_detail, ff.contour_detail)  # local shape differs


def _angularity(contour):
    if len(contour) < 2:
        return 0.0
    return sum(abs(contour[i + 1] - contour[i]) for i in range(len(contour) - 1)) / (len(contour) - 1)


class RoleBiasTests(unittest.TestCase):
    def test_villain_jitter_exceeds_love(self):
        self.assertGreater(_role_contour_jitter("villain"), _role_contour_jitter("love"))

    def test_villain_contour_more_angular_than_love(self):
        # same contour kind, different role: villain's wider jitter -> more angular
        villain = sum(_angularity(_roll_contour("rising", 8, Random(s), "villain")) for s in range(60))
        love = sum(_angularity(_roll_contour("rising", 8, Random(s), "love")) for s in range(60))
        self.assertGreater(villain, love * 1.2)  # deterministic seeds: 10.8 vs 8.4


class M6ExitTests(unittest.TestCase):
    def test_inverted_restatement_registers_as_related(self):
        # bar 2 is the inversion of bar 1; the per-bar signature missed this, M6 detects it.
        kernel = open_kernel(bars=2)
        bar1 = [("C4", 1.0), ("D4", 1.0), ("E4", 1.0), ("F4", 1.0)]
        bar2 = [("C4", 1.0), ("B-3", 1.0), ("A-3", 1.0), ("G-3", 1.0)]  # inversion
        feats = measure_free_features(kernel, bar1 + bar2)
        self.assertGreaterEqual(feats.similarity[0][1], 0.85)

    def test_feature_vector_separates_same_mean_shapes(self):
        kernel = open_kernel(bars=2)
        filler = [("C4", 1.0)] * 4
        rising = [("C4", 1.0), ("E4", 1.0), ("G4", 1.0), ("E4", 1.0)] + filler
        falling = [("E4", 1.0), ("G4", 1.0), ("E4", 1.0), ("C4", 1.0)] + filler
        fr = measure_free_features(kernel, rising)
        ff = measure_free_features(kernel, falling)
        self.assertEqual(fr.contour, ff.contour)  # per-bar means tie
        self.assertNotEqual(_feature_vector(fr), _feature_vector(ff))  # spread space separates them


if __name__ == "__main__":
    unittest.main()
