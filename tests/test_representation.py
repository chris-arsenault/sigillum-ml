import unittest

from music21 import pitch as m21pitch

from generation.theme_nn import decode, encode
from symphony.materials.themes import RENN_FULL, ESHAIA_FULL


def to_midi(items):
    out = []
    for pitch, dur in items:
        out.append((None if pitch is None else int(m21pitch.Pitch(pitch).midi), float(dur)))
    return out


def transpose(items, semis):
    return [(None if m is None else m + semis, d) for m, d in items]


class RepresentationTests(unittest.TestCase):
    def test_roundtrip_is_lossless(self):
        # A real theme with rests -> decode must reproduce it exactly (round-trip is lossless for
        # any declared key; alteration carries non-scale tones).
        melody = to_midi(RENN_FULL)
        self.assertEqual(decode(encode(melody, (5, "major")), (5, "major")), melody)

    def test_roundtrip_with_chromatics(self):
        # A line containing non-scale notes still round-trips (alteration carries them).
        melody = [(60, 1.0), (61, 0.5), (64, 0.5), (66, 1.0), (None, 0.5), (67, 1.0)]
        self.assertEqual(decode(encode(melody, (0, "major")), (0, "major")), melody)

    def test_rest_preserved(self):
        melody = [(60, 1.0), (None, 0.5), (62, 1.0)]
        events = encode(melody, (0, "major"))
        self.assertEqual(events[1].kind, "rest")
        self.assertEqual(decode(events, (0, "major")), melody)

    def test_key_invariance(self):
        # The SAME tune in C and transposed to G encodes to the SAME degree/alteration/duration
        # sequence (only the octave shifts) — the property that lets the model generalise across keys.
        melody_c = to_midi(ESHAIA_FULL)
        melody_g = transpose(melody_c, 7)            # up a perfect fifth
        events_c = encode(melody_c, (0, "major"))    # C major
        events_g = encode(melody_g, (7, "major"))    # G major
        role = lambda e: (e.kind, e.degree, e.alteration, e.duration)
        self.assertEqual([role(e) for e in events_c], [role(e) for e in events_g])

    def test_degree_reads_correctly(self):
        # In F major: C is degree 5, F is degree 1, the tonic.
        events = encode([(60, 1.0), (65, 1.0)], (5, "major"))
        self.assertEqual((events[0].degree, events[0].alteration), (5, 0))   # C = ^5
        self.assertEqual((events[1].degree, events[1].alteration), (1, 0))   # F = ^1


if __name__ == "__main__":
    unittest.main()
