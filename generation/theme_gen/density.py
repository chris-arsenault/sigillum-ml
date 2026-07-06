"""Kernel density / free-space / expected-spread reporting (D5).

Per kernel, compute and surface how constrained the theme is and how wide a candidate
spread is achievable — the tighter the kernel, the narrower the spread, surfaced rather
than hidden. Read-only: nothing here changes generation or selection. Reuses the M3/M4
feasibility machinery (``generation.theme_gen.engine``); engine does not import
this module, so there is no import cycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log2

from generation.theme_gen._common import TICKS_PER_QUARTER, _duration_to_ticks
from generation.theme_gen.engine import (
    _free_segments,
    _lands,
    _legal_pitches,
    _palette_ticks,
)
from generation.theme_gen.kerneldsl import ThemeKernel, _pinned_intervals

# Swappable cap on the realization count (bounds the counting-DP cost and the reported
# spread; counts at or above it are reported as saturated "wide spread").
REALIZATION_CAP = 100_000


@dataclass(frozen=True)
class KernelDensity:
    total_ql: float
    pinned_ql: float
    free_ql: float
    free_space_fraction: float  # 0..1
    density: float  # 1 - free_space_fraction
    n_pitch_pins: int
    n_harmonic_pins: int
    n_structural_pins: int
    feasible_realizations: int = 0  # populated in M5 step 3
    realizations_saturated: bool = False
    expected_spread_bits: float = 0.0


def _segment_realizations(kernel, start, end, next_pin_pitch, cadence_pc, cap):
    """Count legal landing completions of one free segment, capped at ``cap`` (D5).

    Mirrors ``_segment_feasibility`` but sums instead of OR-ing. The count is independent
    of the entering pitch (no interval hard-constraint), so a dummy entry pitch is used.
    """
    end_ticks = _duration_to_ticks(end)
    palette = _palette_ticks(kernel.frame)
    memo: dict[tuple[int, int | None], int] = {}

    def count(remaining: int, pitch: int | None) -> int:
        if remaining == 0:
            return 1 if _lands(pitch, next_pin_pitch, cadence_pc) else 0
        key = (remaining, pitch)
        if key in memo:
            return memo[key]
        offset = (end_ticks - remaining) / TICKS_PER_QUARTER
        legal = _legal_pitches(kernel, offset)
        total = 0
        for duration_ticks in palette:
            if duration_ticks > remaining:
                continue
            for candidate in legal:
                total += count(remaining - duration_ticks, candidate)
                if total >= cap:
                    total = cap
                    break
            if total >= cap:
                break
        memo[key] = total
        return total

    return count(_duration_to_ticks(end - start), None)


def feasible_realizations(kernel: ThemeKernel, cap: int = REALIZATION_CAP) -> tuple[int, bool]:
    """(capped count of feasible candidate realizations, saturated?) for the kernel (D5).

    Product of per-free-segment completion counts; a fully pinned kernel has exactly one.
    Returns ``(0, False)`` if the kernel is infeasible.
    """
    total = 1
    saturated = False
    for start, end, next_pin_pitch, cadence_pc in _free_segments(kernel):
        segment_count = _segment_realizations(kernel, start, end, next_pin_pitch, cadence_pc, cap)
        if segment_count == 0:
            return 0, False
        total *= segment_count
        if total >= cap:
            total = cap
            saturated = True
            break
    return total, saturated


def kernel_density(kernel: ThemeKernel, cap: int = REALIZATION_CAP) -> KernelDensity:
    frame = kernel.frame
    total_ql = frame.total_ql
    pinned_ql = sum(end - start for start, end in _pinned_intervals(kernel))
    free_ql = max(0.0, total_ql - pinned_ql)
    free_fraction = free_ql / total_ql if total_ql > 0 else 0.0
    realizations, saturated = feasible_realizations(kernel, cap)
    return KernelDensity(
        total_ql=total_ql,
        pinned_ql=pinned_ql,
        free_ql=free_ql,
        free_space_fraction=free_fraction,
        density=1.0 - free_fraction,
        n_pitch_pins=len(kernel.pitch_rhythm_pins),
        n_harmonic_pins=len(kernel.harmonic_pins),
        n_structural_pins=len(kernel.structural_pins),
        feasible_realizations=realizations,
        realizations_saturated=saturated,
        expected_spread_bits=log2(realizations) if realizations > 0 else 0.0,
    )
