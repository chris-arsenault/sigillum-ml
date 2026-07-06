"""Kernel vocabulary and frame/kernel geometry.

The Stage-0 kernel dataclasses (the shared obligation envelope) plus the pure helpers that
read frame/kernel structure: pin offsets, free-bar computation, scale/range geometry, and
kernel validation. This module carries the original prototype's kernel layer forward
(plan M0).
Decision D4 (``docs/architecture/17_theme_generator.md``): kernels are authored in the
repo's item-list DSL; the kernel-authoring helper layer (M7) will live alongside this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from music21 import pitch as m21pitch

from generation.theme_gen._common import (
    EPS,
    MelodyItem,
    _first_pitch,
    _normalize_item,
    _round_duration,
)


@dataclass(frozen=True)
class ThemeFrame:
    """Fixed thematic frame shared by every candidate in a batch."""

    bars: int
    meter: str = "4/4"
    beats_per_bar: float = 4.0
    key: str = "C"
    scale: tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")
    role: str = "neutral"
    lower: str = "C4"
    upper: str = "C6"
    duration_palette: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

    @property
    def total_ql(self) -> float:
        return self.bars * self.beats_per_bar


@dataclass(frozen=True)
class PitchRhythmPin:
    """Specific pitch/rhythm material that must occur at a fixed location."""

    bar: int
    beat: float
    items: tuple[MelodyItem, ...]
    label: str = ""

    @property
    def name(self) -> str:
        return self.label or f"pitch-rhythm pin b{self.bar}:{self.beat:g}"


@dataclass(frozen=True)
class HarmonicPin:
    """A required harmonic region for a bar.

    `chord_tones` is optional. When supplied, generation biases strong starts toward
    these pitch classes and `pins_satisfied` can confirm at least one matching tone in
    the bar. Empty `chord_tones` keeps the harmonic pin as auditable metadata.
    """

    bar: int
    label: str
    chord_tones: tuple[str, ...] = ()


@dataclass(frozen=True)
class StructuralPin:
    """Phrase/cadence architecture metadata for the kernel contract."""

    start_bar: int
    end_bar: int
    label: str
    cadence: str = ""


@dataclass(frozen=True)
class ThemeKernel:
    """The shared obligation envelope for a generated candidate batch."""

    frame: ThemeFrame
    pitch_rhythm_pins: tuple[PitchRhythmPin, ...] = ()
    harmonic_pins: tuple[HarmonicPin, ...] = ()
    structural_pins: tuple[StructuralPin, ...] = ()


# ---- Kernel-authoring DSL (D4) ---------------------------------------------------------
# Terse constructors over the item-list DSL so kernels are authored in small Python files
# (no JSON/YAML). Pure sugar over the dataclasses above.


def frame(
    bars: int,
    *,
    key: str = "C",
    role: str = "neutral",
    meter: str = "4/4",
    beats_per_bar: float = 4.0,
    scale: Sequence[str] = ("C", "D", "E", "F", "G", "A", "B"),
    lower: str = "C4",
    upper: str = "C6",
    durations: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
) -> ThemeFrame:
    return ThemeFrame(
        bars=bars,
        meter=meter,
        beats_per_bar=beats_per_bar,
        key=key,
        scale=tuple(scale),
        role=role,
        lower=lower,
        upper=upper,
        duration_palette=tuple(durations),
    )


def pin(bar: int, beat: float, *items, label: str = "") -> PitchRhythmPin:
    """A pitch/rhythm pin from item-list tuples, e.g. ``pin(1, 1.0, ("E5", 1.5), ("B4", 0.5))``."""
    return PitchRhythmPin(bar=bar, beat=beat, items=tuple(items), label=label)


def harm(bar: int, label: str, *chord_tones: str) -> HarmonicPin:
    return HarmonicPin(bar=bar, label=label, chord_tones=tuple(chord_tones))


def phrase(start_bar: int, end_bar: int, label: str, *, cadence: str = "") -> StructuralPin:
    return StructuralPin(start_bar=start_bar, end_bar=end_bar, label=label, cadence=cadence)


def kernel(frame: ThemeFrame, *, pins=(), harmony=(), structure=()) -> ThemeKernel:
    return ThemeKernel(
        frame=frame,
        pitch_rhythm_pins=tuple(pins),
        harmonic_pins=tuple(harmony),
        structural_pins=tuple(structure),
    )


def _validate_kernel(kernel: ThemeKernel) -> None:
    frame = kernel.frame
    if frame.bars <= 0 or frame.beats_per_bar <= 0:
        raise ValueError("theme frame must have positive bars and beats_per_bar")
    intervals = _pinned_intervals(kernel)
    for start, end in intervals:
        if start < -EPS or end - frame.total_ql > EPS:
            raise ValueError("pitch/rhythm pin lies outside the frame")
    for first, second in zip(intervals, intervals[1:]):
        if first[1] - second[0] > EPS:
            raise ValueError("pitch/rhythm pins overlap")


def _pin_timed_items(kernel: ThemeKernel) -> list[tuple[float, tuple[MelodyItem, ...]]]:
    pins = []
    for pin in kernel.pitch_rhythm_pins:
        pins.append((_pin_offset(kernel.frame, pin), tuple(_normalize_item(item) for item in pin.items)))
    return sorted(pins, key=lambda pair: pair[0])


def _pinned_intervals(kernel: ThemeKernel) -> list[tuple[float, float]]:
    intervals = []
    for offset, items in _pin_timed_items(kernel):
        intervals.append((offset, offset + sum(float(item[1]) for item in items)))
    return sorted(intervals)


def _pin_offset(frame: ThemeFrame, pin: PitchRhythmPin) -> float:
    return (pin.bar - 1) * frame.beats_per_bar + (pin.beat - 1)


def _frame_bounds(frame: ThemeFrame) -> tuple[int, int]:
    return int(m21pitch.Pitch(frame.lower).midi), int(m21pitch.Pitch(frame.upper).midi)


def _scale_pitch_names(frame: ThemeFrame) -> tuple[str, ...]:
    return tuple(name[:-1] if name[-1:].isdigit() else name for name in frame.scale)


def _pitch_classes(names: Sequence[str]) -> set[int]:
    pcs = set()
    for name in names:
        pitch_name = name if name[-1:].isdigit() else name + "4"
        pcs.add(int(m21pitch.Pitch(pitch_name).pitchClass))
    return pcs


def _scale_midis(frame: ThemeFrame) -> list[int]:
    low, high = _frame_bounds(frame)
    pcs = _pitch_classes(_scale_pitch_names(frame))
    return [midi for midi in range(low, high + 1) if midi % 12 in pcs]


def _scale_pcs(frame: ThemeFrame) -> list[int]:
    """Pitch classes of the frame's scale, in scale order (M4 harmony geometry)."""
    return [int(m21pitch.Pitch(name).pitchClass) for name in _scale_pitch_names(frame)]


def _tonic_pc(frame: ThemeFrame) -> int:
    return int(m21pitch.Pitch(frame.key).pitchClass)


def _dominant_pc(frame: ThemeFrame) -> int:
    return (_tonic_pc(frame) + 7) % 12


def _scale_degree(frame: ThemeFrame, pitch_class: int) -> int | None:
    """1-based scale degree of ``pitch_class``, or None if chromatic to the scale."""
    pcs = _scale_pcs(frame)
    return pcs.index(pitch_class) + 1 if pitch_class in pcs else None


def _diatonic_triads(frame: ThemeFrame) -> dict[int, frozenset[int]]:
    """Scale-degree -> triad pitch classes (thirds stacked within the scale)."""
    pcs = _scale_pcs(frame)
    n = len(pcs)
    return {
        i + 1: frozenset({pcs[i], pcs[(i + 2) % n], pcs[(i + 4) % n]})
        for i in range(n)
    }


def _cadence_target_pc(frame: ThemeFrame, cadence: str) -> int | None:
    """Cadence target pitch class: authentic/plagal -> tonic, half -> dominant, else None."""
    text = cadence.lower()
    if any(word in text for word in ("authentic", "perfect", "plagal")):
        return _tonic_pc(frame)
    if "half" in text or "semicadence" in text:
        return _dominant_pc(frame)
    return None


def _spell_midi(frame: ThemeFrame, midi: int) -> str:
    for name in _scale_pitch_names(frame):
        for octave in range(-1, 10):
            try:
                pitch = m21pitch.Pitch(f"{name}{octave}")
            except Exception:
                continue
            if int(pitch.midi) == midi:
                return pitch.nameWithOctave
    pitch = m21pitch.Pitch()
    pitch.midi = midi
    return pitch.nameWithOctave


def _initial_pitch(frame: ThemeFrame, pins: Sequence[tuple[float, Sequence[MelodyItem]]]) -> int:
    first = _first_pitch(pins[0][1]) if pins else None
    if first is not None:
        return first
    low, high = _frame_bounds(frame)
    return (low + high) // 2


def _duration_palette(frame: ThemeFrame) -> tuple[float, ...]:
    return tuple(sorted({_round_duration(duration) for duration in frame.duration_palette}))


# Bar-relative strong-beat fractions (M4): chord tones are required on these beats of a
# harmonic-pinned bar. Single source for both the generator and the pins_satisfied filter.
HARMONIC_STRONG_BEAT_FRACTIONS = (0.0, 0.5)


def _is_strong_beat(frame: ThemeFrame, offset: float) -> bool:
    beat_in_bar = offset % frame.beats_per_bar
    return any(
        abs(beat_in_bar - fraction * frame.beats_per_bar) < EPS
        for fraction in HARMONIC_STRONG_BEAT_FRACTIONS
    )


def _harmonic_pcs_for_offset(kernel: ThemeKernel, offset: float) -> set[int]:
    frame = kernel.frame
    if not _is_strong_beat(frame, offset):
        return set()
    bar = _bar_index(frame, offset) + 1
    for pin in kernel.harmonic_pins:
        if pin.bar == bar and pin.chord_tones:
            return _pitch_classes(pin.chord_tones)
    return set()


def _bar_index(frame: ThemeFrame, offset: float) -> int:
    return int(min(frame.bars - 1, max(0, offset // frame.beats_per_bar)))


def _free_bars(kernel: ThemeKernel) -> tuple[int, ...]:
    pinned = _pinned_intervals(kernel)
    bars = []
    for index in range(kernel.frame.bars):
        start = index * kernel.frame.beats_per_bar
        end = start + kernel.frame.beats_per_bar
        if any(max(start, p0) < min(end, p1) - EPS for p0, p1 in pinned):
            if end - start > sum(max(0.0, min(end, p1) - max(start, p0)) for p0, p1 in pinned) + EPS:
                bars.append(index + 1)
        else:
            bars.append(index + 1)
    return tuple(bars)


def _offset_inside_any(offset: float, intervals: Sequence[tuple[float, float]]) -> bool:
    return any(start - EPS <= offset < end - EPS for start, end in intervals)
