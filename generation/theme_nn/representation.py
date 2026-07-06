"""The note representation for the neural theme model — describe a note by its ROLE in the key.

A melody is a list of ``(midi | None, duration)`` items (``None`` = rest). ``encode`` turns each
note into an ``Event`` whose pitch is its scale degree (1-7) + a chromatic alteration + octave,
relative to the segment's key; ``decode`` inverts it exactly. Two properties matter and are tested:

  * **lossless** — decode(encode(x)) == x (the representation never corrupts the music), and
  * **key-invariant** — the same tune in any key encodes to the SAME degree/alteration/duration
    sequence (only the octave shifts). That invariance is the whole point: the model learns a
    pattern once and it applies in every key, which is what lets a small corpus go far.

This is the shared foundation: a flat note-predictor and an explicit motif-developer (the deferred
operational layer) would both build on exactly this.
"""
from __future__ import annotations

from dataclasses import dataclass

from framework.analysis.tonal import SCALE_OFFSETS

MelodyItem = tuple[int | None, float]


@dataclass(frozen=True)
class Event:
    kind: str              # "note" | "rest"
    degree: int | None     # scale degree 1-7 (None for a rest)
    alteration: int        # semitones above the degree: 0 diatonic, 1-2 for a chromatic note
    octave: int            # absolute octave (midi // 12); 0 for a rest
    duration: float


def _encode_pitch(midi: int, tonic: int, offsets: tuple[int, ...]) -> tuple[int, int, int]:
    rel = (midi % 12 - tonic) % 12                 # semitones above the tonic, 0-11
    base = max(o for o in offsets if o <= rel)     # the scale degree at or below it
    return offsets.index(base) + 1, rel - base, midi // 12


def encode(melody: list[MelodyItem], key: tuple[int, str]) -> list[Event]:
    tonic, mode = key
    offsets = SCALE_OFFSETS[mode]
    out: list[Event] = []
    for pitch, duration in melody:
        if pitch is None:
            out.append(Event("rest", None, 0, 0, float(duration)))
        else:
            degree, alteration, octave = _encode_pitch(int(pitch), tonic, offsets)
            out.append(Event("note", degree, alteration, octave, float(duration)))
    return out


def decode(events: list[Event], key: tuple[int, str]) -> list[MelodyItem]:
    tonic, mode = key
    offsets = SCALE_OFFSETS[mode]
    out: list[MelodyItem] = []
    for event in events:
        if event.kind == "rest":
            out.append((None, event.duration))
        else:
            rel = offsets[event.degree - 1] + event.alteration
            midi = event.octave * 12 + (tonic + rel) % 12
            out.append((midi, event.duration))
    return out
