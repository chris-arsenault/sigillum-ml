"""Stage-2 randomized targets and free-region feature measurement.

Rolls a per-candidate random target profile (contour / surprise / self-similarity),
measures the candidate's free regions (pinned regions subtracted), and scores the distance
between them. Decisions D3 (surprise meter) and the M6 milestone deepen the features; the
spread space they define stays the basis of Stage-3 selection. This module carries the
original prototype's feature measurement forward (plan M0).
"""
from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Sequence

from generation.theme_gen._common import (
    EPS,
    MelodyItem,
    _clamp,
    _item_midi,
    _matrix_distance,
    _normalize_vector,
    _round_duration,
    _TimedItem,
    _time_items,
    _vector_distance,
    _weighted_choice,
)
from generation.theme_gen.corpus import default_theme_corpus
from generation.theme_gen.kerneldsl import (
    ThemeKernel,
    _bar_index,
    _frame_bounds,
    _free_bars,
    _offset_inside_any,
    _pinned_intervals,
)
from generation.theme_gen.model import MarkovMelodyModel


@dataclass(frozen=True)
class TargetProfile:
    contour_kind: str
    contour: tuple[float, ...]
    surprise_peak_bar: int | None
    surprise: tuple[float, ...]
    similarity: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class FeatureSummary:
    contour: tuple[float, ...]
    surprise: tuple[float, ...]
    similarity: tuple[tuple[float, ...], ...]
    free_bars: tuple[int, ...]
    contour_detail: tuple[float, ...] = ()  # per free bar (first, last) normalized pitch (M6)


@dataclass(frozen=True)
class ConformanceWeights:
    contour: float = 1.0
    surprise: float = 0.25
    similarity: float = 0.85


def roll_target_profile(kernel: ThemeKernel, rng: Random) -> TargetProfile:
    frame = kernel.frame
    role = frame.role.lower()
    contour_kind = _weighted_role_choice(
        rng,
        role,
        {
            "love": {"arch": 4, "late_peak": 4, "rising": 2, "valley": 1, "terraced": 1},
            "heroic": {"rising": 4, "arch": 3, "late_peak": 2, "terraced": 1, "valley": 1},
            "villain": {"falling": 3, "valley": 3, "terraced": 3, "late_peak": 1, "arch": 1},
        },
        ("rising", "falling", "arch", "valley", "late_peak", "terraced"),
    )
    contour = _roll_contour(contour_kind, frame.bars, rng, role)
    free_bars = _free_bars(kernel)
    surprise_peak = _roll_surprise_peak(free_bars, role, rng)
    surprise = _surprise_curve(frame.bars, surprise_peak)
    similarity = _roll_similarity_target(frame.bars, free_bars, rng)
    return TargetProfile(contour_kind, contour, surprise_peak, surprise, similarity)


def measure_free_features(
    kernel: ThemeKernel,
    items: Sequence[MelodyItem],
    model: MarkovMelodyModel | None = None,
) -> FeatureSummary:
    model = model or MarkovMelodyModel.from_corpus(default_theme_corpus())
    frame = kernel.frame
    timed = _time_items(items)
    pinned = _pinned_intervals(kernel)
    low, high = _frame_bounds(frame)
    by_bar: list[list[_TimedItem]] = [[] for _ in range(frame.bars)]
    free_bars: set[int] = set()

    previous_pitch = None
    previous_interval = 0
    previous_duration = None
    surprises: list[list[float]] = [[] for _ in range(frame.bars)]
    for timed_item in timed:
        midi = _item_midi(timed_item.item)
        if midi is not None and previous_pitch is not None:
            current_interval = midi - previous_pitch
        else:
            current_interval = previous_interval

        bar_index = _bar_index(frame, timed_item.offset)
        is_pinned = _offset_inside_any(timed_item.offset, pinned)
        if not is_pinned and 0 <= bar_index < frame.bars:
            by_bar[bar_index].append(timed_item)
            free_bars.add(bar_index + 1)
            surprises[bar_index].append(
                model.transition_surprise(
                    previous_pitch, previous_interval, previous_duration, timed_item.item
                )
            )

        if midi is not None:
            if previous_pitch is not None:
                previous_interval = current_interval
            previous_pitch = midi
        previous_duration = _round_duration(float(timed_item.item[1]))

    contour_values = []
    detail_values = []
    surprise_values = []
    span = max(high - low, 1)
    for index, bar_items in enumerate(by_bar):
        pitches = [_item_midi(t.item) for t in bar_items if _item_midi(t.item) is not None]
        if pitches:
            avg = sum(pitches) / len(pitches)
            contour_values.append(_clamp((avg - low) / span, 0.0, 1.0))
            detail_values.append(_clamp((pitches[0] - low) / span, 0.0, 1.0))   # arrival
            detail_values.append(_clamp((pitches[-1] - low) / span, 0.0, 1.0))  # departure
        else:
            contour_values.append(0.0)
            detail_values.extend((0.0, 0.0))

        if surprises[index]:
            surprise_values.append(sum(surprises[index]) / len(surprises[index]))
        else:
            surprise_values.append(0.0)
    surprise_values = _normalize_vector(surprise_values)
    similarity = _measure_similarity(by_bar)

    return FeatureSummary(
        tuple(contour_values),
        tuple(surprise_values),
        tuple(tuple(row) for row in similarity),
        tuple(sorted(free_bars)),
        tuple(detail_values),
    )


def conformance_distance(
    features: FeatureSummary,
    target: TargetProfile,
    weights: ConformanceWeights = ConformanceWeights(),
) -> float:
    contour = _vector_distance(features.contour, target.contour)
    surprise = _vector_distance(features.surprise, target.surprise)
    similarity = _matrix_distance(features.similarity, target.similarity)
    total_weight = weights.contour + weights.surprise + weights.similarity
    if total_weight <= EPS:
        return 0.0
    weighted = (
        weights.contour * contour
        + weights.surprise * surprise
        + weights.similarity * similarity
    )
    return weighted / total_weight


def _weighted_role_choice(
    rng: Random,
    role: str,
    presets: dict[str, dict[str, float]],
    fallback: Sequence[str],
) -> str:
    weights = presets.get(role)
    if not weights:
        return rng.choice(tuple(fallback))
    choices = tuple(weights)
    return _weighted_choice(choices, [weights[choice] for choice in choices], rng)


# M6 role-bias contour bounds (swappable): per-role jitter amount applied to the rolled
# macro-shape. Villain leans angular (high jitter), love gentle (low jitter); role shapes
# roll bounds only, it never pins notes.
_ROLE_CONTOUR = {
    "villain": {"jitter": 0.22},
    "heroic": {"jitter": 0.10},
    "love": {"jitter": 0.04},
}
_DEFAULT_CONTOUR_JITTER = 0.08


def _role_contour_jitter(role: str) -> float:
    return _ROLE_CONTOUR.get(role.lower(), {}).get("jitter", _DEFAULT_CONTOUR_JITTER)


def _roll_contour(kind: str, bars: int, rng: Random, role: str = "neutral") -> tuple[float, ...]:
    if bars == 1:
        return (0.5,)
    xs = [i / (bars - 1) for i in range(bars)]
    if kind == "rising":
        values = xs
    elif kind == "falling":
        values = [1.0 - x for x in xs]
    elif kind == "arch":
        values = [1.0 - abs((x - 0.5) * 2.0) for x in xs]
    elif kind == "valley":
        values = [abs((x - 0.5) * 2.0) for x in xs]
    elif kind == "late_peak":
        peak = rng.uniform(0.62, 0.88)
        values = [x / peak if x <= peak else max(0.0, 1.0 - (x - peak) / (1.0 - peak)) for x in xs]
    elif kind == "terraced":
        split = rng.randint(2, max(2, bars - 2))
        left = rng.uniform(0.2, 0.55)
        right = rng.uniform(0.55, 0.9)
        values = [left if i < split else right for i in range(bars)]
    else:
        values = [0.5 for _ in xs]
    jitter = _role_contour_jitter(role)
    jittered = [_clamp(value + rng.uniform(-jitter, jitter), 0.0, 1.0) for value in values]
    return tuple(jittered)


def _roll_surprise_peak(free_bars: Sequence[int], role: str, rng: Random) -> int | None:
    if not free_bars:
        return None
    bars = list(free_bars)
    if role == "love" and len(bars) > 2:
        later = bars[len(bars) // 2 :]
        return rng.choice(later)
    if role == "villain" and len(bars) > 2:
        return _weighted_choice(bars, [1.3 if i % 2 == 0 else 1.0 for i, _ in enumerate(bars)], rng)
    return rng.choice(bars)


def _surprise_curve(bars: int, peak_bar: int | None) -> tuple[float, ...]:
    if peak_bar is None:
        return tuple(0.0 for _ in range(bars))
    values = []
    for bar in range(1, bars + 1):
        distance = abs(bar - peak_bar)
        if distance == 0:
            values.append(1.0)
        elif distance == 1:
            values.append(0.45)
        else:
            values.append(0.12)
    return tuple(values)


def _roll_similarity_target(bars: int, free_bars: Sequence[int], rng: Random) -> tuple[tuple[float, ...], ...]:
    matrix = [[0.18 for _ in range(bars)] for _ in range(bars)]
    for i in range(bars):
        matrix[i][i] = 1.0
    free = list(free_bars)
    rng.shuffle(free)
    pair_count = min(max(1, len(free) // 3), len(free) // 2) if len(free) >= 2 else 0
    for index in range(pair_count):
        a = free[index * 2] - 1
        b = free[index * 2 + 1] - 1
        value = rng.uniform(0.62, 0.82)
        matrix[a][b] = value
        matrix[b][a] = value
    return tuple(tuple(row) for row in matrix)


def _measure_similarity(by_bar: Sequence[Sequence[_TimedItem]]) -> list[list[float]]:
    features = [_bar_signature(items) for items in by_bar]
    size = len(features)
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i == j:
                matrix[i][j] = 1.0
            else:
                matrix[i][j] = _signature_similarity(features[i], features[j])
    return matrix


def _bar_signature(items: Sequence[_TimedItem]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    pitches = [_item_midi(item.item) for item in items if _item_midi(item.item) is not None]
    intervals = tuple(pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1))
    rhythms = tuple(_round_duration(float(item.item[1])) for item in items)
    return intervals, rhythms


# M6 self-similarity strictness (swappable): which motivic transforms count as recurrence,
# and how much a partial-cell (sub-run) match is worth relative to a full match.
SIMILARITY_DETECT_INVERSION = True
SIMILARITY_DETECT_SCALING = True
PARTIAL_CELL_WEIGHT = 0.8


def _scale_normalize(rhythms: Sequence[float]) -> tuple[float, ...]:
    """Scale-invariant rhythm sequence (augmentation/diminution collapse to one shape)."""
    if not rhythms or rhythms[0] == 0:
        return tuple(rhythms)
    return tuple(value / rhythms[0] for value in rhythms)


def _partial_cell(left: Sequence[int], right: Sequence[int]) -> float:
    """Longest common contiguous interval sub-run as a fraction (partial-cell recurrence)."""
    if not left or not right:
        return 0.0
    best = 0
    for i in range(len(left)):
        for j in range(len(right)):
            k = 0
            while i + k < len(left) and j + k < len(right) and left[i + k] == right[j + k]:
                k += 1
            best = max(best, k)
    if best < 2:
        return 0.0
    return best / max(len(left), len(right))


def _signature_similarity(
    left: tuple[tuple[int, ...], tuple[float, ...]],
    right: tuple[tuple[int, ...], tuple[float, ...]],
) -> float:
    if not left[0] and not left[1] and not right[0] and not right[1]:
        return 1.0
    if not left[1] or not right[1]:
        return 0.0

    inversions = (False, True) if SIMILARITY_DETECT_INVERSION else (False,)
    best = 0.0
    for invert in inversions:
        right_intervals = tuple(-x for x in right[0]) if invert else right[0]
        interval_score = _sequence_similarity(left[0], right_intervals, tolerance=2.0)
        rhythm_score = _sequence_similarity(left[1], right[1], tolerance=0.5)
        if SIMILARITY_DETECT_SCALING:
            scaled = _sequence_similarity(
                _scale_normalize(left[1]), _scale_normalize(right[1]), tolerance=0.25
            )
            rhythm_score = max(rhythm_score, scaled)
        best = max(best, 0.55 * interval_score + 0.45 * rhythm_score)

    partial = _partial_cell(left[0], right[0])
    return max(best, PARTIAL_CELL_WEIGHT * partial)


def _sequence_similarity(left: Sequence[float], right: Sequence[float], tolerance: float) -> float:
    length = max(len(left), len(right))
    if length == 0:
        return 1.0
    score = 0.0
    for i in range(length):
        if i >= len(left) or i >= len(right):
            continue
        score += max(0.0, 1.0 - abs(float(left[i]) - float(right[i])) / tolerance)
    return score / length


def _feature_vector(features: FeatureSummary) -> tuple[float, ...]:
    values: list[float] = (
        list(features.contour) + list(features.surprise) + list(features.contour_detail)
    )
    for row_index, row in enumerate(features.similarity):
        values.extend(row[row_index + 1 :])
    return tuple(values)
