"""Frame-agnostic primitives shared across the theme_gen package.

Constants, the timed-item record, and the timing / math / item-pitch helpers that carry no
knowledge of the kernel or frame. Lowest layer of the package import graph: this module
imports nothing from its siblings. This holds the shared helpers from the original
prototype (plan M0, decision to host shared helpers here rather than in ``kerneldsl``).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from random import Random
from typing import Any, Sequence

from music21 import pitch as m21pitch

MelodyItem = tuple[Any, ...]
TICKS_PER_QUARTER = 12
EPS = 1e-6


@dataclass(frozen=True)
class _TimedItem:
    offset: float
    item: MelodyItem
    pinned: bool = False

    @property
    def end(self) -> float:
        return self.offset + float(self.item[1])


def _time_items(items: Sequence[MelodyItem]) -> list[_TimedItem]:
    timed: list[_TimedItem] = []
    offset = 0.0
    for item in items:
        normalized = _normalize_item(item)
        timed.append(_TimedItem(offset, normalized))
        offset += float(normalized[1])
    return timed


def _normalize_item(item: MelodyItem) -> MelodyItem:
    if len(item) < 2:
        raise ValueError(f"bad melody item: {item!r}")
    return tuple(item)


def _item_midi(item: MelodyItem) -> int | None:
    value = item[0]
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    return int(m21pitch.Pitch(value).midi)


def _first_pitch(items: Sequence[MelodyItem]) -> int | None:
    for item in items:
        midi = _item_midi(item)
        if midi is not None:
            return midi
    return None


def _same_pitch_duration(left: MelodyItem, right: MelodyItem) -> bool:
    return left[0] == right[0] and abs(float(left[1]) - float(right[1])) < EPS


def _pitch_key(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _round_duration(value: float) -> float:
    return round(value, 6)


def _duration_to_ticks(duration: float) -> int:
    return int(round(duration * TICKS_PER_QUARTER))


def _tick(offset: float) -> int:
    return _duration_to_ticks(offset)


def _ticks_fillable(ticks: int, palette: tuple[float, ...]) -> bool:
    palette_ticks = tuple(sorted({_duration_to_ticks(duration) for duration in palette}))

    @lru_cache(maxsize=None)
    def fillable(remaining: int) -> bool:
        if remaining == 0:
            return True
        if remaining < 0:
            return False
        return any(fillable(remaining - tick) for tick in palette_ticks if tick > 0)

    return fillable(ticks)


def _vector_distance(left: Sequence[float], right: Sequence[float]) -> float:
    length = max(len(left), len(right))
    if length == 0:
        return 0.0
    total = 0.0
    for index in range(length):
        a = left[index] if index < len(left) else 0.0
        b = right[index] if index < len(right) else 0.0
        total += (a - b) ** 2
    return sqrt(total / length)


def _matrix_distance(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    if not left and not right:
        return 0.0
    values_left = [value for row in left for value in row]
    values_right = [value for row in right for value in row]
    return _vector_distance(values_left, values_right)


def _normalize_vector(values: Sequence[float]) -> tuple[float, ...]:
    if not values:
        return ()
    high = max(values)
    low = min(values)
    if high - low < EPS:
        return tuple(0.0 for _ in values)
    return tuple((value - low) / (high - low) for value in values)


def _counter_probability(counter: Counter[Any], value: Any, smoothing: float = 0.25) -> float:
    total = sum(counter.values())
    vocab = max(len(counter), 1)
    return (counter.get(value, 0.0) + smoothing) / (total + smoothing * (vocab + 1))


def _weighted_choice(choices: Sequence[Any], weights: Sequence[float], rng: Random) -> Any:
    total = sum(max(weight, 0.0) for weight in weights)
    if total <= EPS:
        return rng.choice(tuple(choices))
    roll = rng.random() * total
    acc = 0.0
    for choice, weight in zip(choices, weights):
        acc += max(weight, 0.0)
        if acc >= roll:
            return choice
    return choices[-1]


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))
