"""Stage-1 constrained generation, pin verification, and Stage-3 spread selection.

Places the pinned material first, generates connective material toward each next pin, and
greedily selects the most mutually-dissimilar batch. Decision D2
(``docs/architecture/17_theme_generator.md``) replaces the greedy fill with a hybrid
feasibility-gated + bounded-backtracking engine (plan M3); pin enforcement strengthens in
M4. This module carries the original prototype's generation engine forward (plan M0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from random import Random
from typing import Any, Iterable, Sequence

from generation.theme_gen._common import (
    EPS,
    MelodyItem,
    TICKS_PER_QUARTER,
    _duration_to_ticks,
    _first_pitch,
    _item_midi,
    _normalize_item,
    _pitch_key,
    _round_duration,
    _same_pitch_duration,
    _tick,
    _time_items,
    _vector_distance,
    _weighted_choice,
)
from generation.theme_gen.corpus import default_theme_corpus
from generation.theme_gen.features import (
    ConformanceWeights,
    FeatureSummary,
    TargetProfile,
    _feature_vector,
    conformance_distance,
    measure_free_features,
    roll_target_profile,
)
from generation.theme_gen.kerneldsl import (
    HARMONIC_STRONG_BEAT_FRACTIONS,
    ThemeFrame,
    ThemeKernel,
    _cadence_target_pc,
    _diatonic_triads,
    _duration_palette,
    _frame_bounds,
    _harmonic_pcs_for_offset,
    _initial_pitch,
    _pin_offset,
    _pin_timed_items,
    _pitch_classes,
    _scale_midis,
    _spell_midi,
    _validate_kernel,
)
from generation.theme_gen.model import (
    Context,
    MarkovMelodyModel,
    _accent_bucket,
    _harmonic_function_bucket,
    _interval_bucket,
)

# D2 tunables (swappable): the widest leap a segment's last note may make onto the next
# pin, and the per-candidate backtracking budget before a candidate is abandoned.
MAX_LANDING_LEAP = 12
MAX_BACKTRACKS = 64

# Harmonic-implication strictness (swappable): the strong beats whose onsets must be chord
# tones, shared with the generator via kerneldsl.HARMONIC_STRONG_BEAT_FRACTIONS so the
# filter and the generated material stay in lockstep.
_HARMONIC_STRONG_BEAT_FRACTIONS = HARMONIC_STRONG_BEAT_FRACTIONS

# M4 cadence strictness (swappable): when True, also require a stepwise / 4th-5th approach
# into the cadence note; default False checks only the cadential scale degree.
CADENCE_CHECK_APPROACH = False

# Soft-pin scoring (swappable): harmonic-implication and cadence-articulation misses no longer
# *drop* a candidate. That hard gate discarded ~85% of generated material and collapsed the
# spread toward the conventional few that happened to imply the chord. Each soft miss instead
# adds this much to the conformance distance, so spread selection still leans toward
# harmonically-apt candidates while the un-conforming majority keeps the variety alive.
SOFT_PIN_PENALTY = 0.06

# Motif reuse (D7), OFF BY DEFAULT: when enabled, at a free bar-start the generator may quote the
# preceding generated bar's cell (interval+rhythm shape, transposed to continue from the current
# pitch), replayed as a strongly-biased path through the feasibility-gated sampler. It is an
# explicit, deliberate override of the corpus's own continuation, so it stays off unless asked
# for. ECHO_BIAS is how hard a quoted move is preferred when feasible.
ECHO_ENABLED = False
ECHO_BIAS = 1000.0

# Per-bar quote probability is the mean off-diagonal of a candidate's rolled similarity plan,
# scaled by ECHO_RATE_GAIN (the raw mean sits low ~0.2, too quiet to be heard) and capped by
# ECHO_RATE_CAP so even an echo-dense plan never collapses the line to verbatim repetition.
# Many literal quotes still break on range/harmony feasibility, so the realized repeat rate is
# well below this; raise the gain/cap to make motifs recur more insistently.
ECHO_RATE_GAIN = 1.8
ECHO_RATE_CAP = 0.5

# Two-level steering (swappable, D7): each candidate rolls its own per-bar contour plan, and
# free-material pitch choices are softly pulled toward the planned register for the bar — so the
# line is *generated toward* a shape instead of only scored against one afterward. The pull is a
# soft weight multiplier (per-candidate plans stay randomized, so the spread is preserved, never
# a shared target). 0 disables steering; larger tightens the line to the plan.
STEER_STRENGTH = 1.2


def _span_sounding_midis(frame: ThemeFrame, start_bar: int, end_bar: int, timed) -> list[int]:
    span_start = (start_bar - 1) * frame.beats_per_bar
    span_end = end_bar * frame.beats_per_bar
    return [
        _item_midi(entry.item)
        for entry in timed
        if _item_midi(entry.item) is not None and span_start - EPS <= entry.offset < span_end - EPS
    ]


def _cadence_articulated(frame: ThemeFrame, pin, timed) -> bool:
    """Whether the structural span articulates its declared cadence (M4).

    The span's final sounding note must be the cadence target scale degree (authentic ->
    tonic, half -> dominant); unrecognized/empty cadence is metadata and always passes.
    """
    target = _cadence_target_pc(frame, pin.cadence)
    if target is None:
        return True
    midis = _span_sounding_midis(frame, pin.start_bar, pin.end_bar, timed)
    if not midis or midis[-1] % 12 != target:
        return False
    if CADENCE_CHECK_APPROACH and len(midis) >= 2:
        interval = abs(midis[-1] - midis[-2])
        if interval > 2 and interval not in (5, 7):
            return False
    return True


def _strong_beat_offsets(frame: ThemeFrame, bar: int) -> list[float]:
    base = (bar - 1) * frame.beats_per_bar
    return [base + fraction * frame.beats_per_bar for fraction in _HARMONIC_STRONG_BEAT_FRACTIONS]


def _harmonic_implies(frame: ThemeFrame, pin, timed) -> bool:
    """Whether the bar implies the declared chord (M4): strong-beat onsets are chord tones
    and no competing diatonic triad matches the bar's content more strongly."""
    chord_pcs = _pitch_classes(pin.chord_tones)
    base = (pin.bar - 1) * frame.beats_per_bar
    end = base + frame.beats_per_bar
    strong_offsets = _strong_beat_offsets(frame, pin.bar)

    saw_strong = False
    bar_notes: list[tuple[int, int]] = []  # (pitch class, weight)
    for entry in timed:
        midi = _item_midi(entry.item)
        if midi is None or not (base - EPS <= entry.offset < end - EPS):
            continue
        is_strong = any(abs(entry.offset - offset) < EPS for offset in strong_offsets)
        if is_strong:
            saw_strong = True
            if midi % 12 not in chord_pcs:
                return False
        bar_notes.append((midi % 12, 2 if is_strong else 1))

    if not saw_strong:
        return False

    triads = _diatonic_triads(frame)
    declared = frozenset(chord_pcs)

    def score(triad: frozenset[int]) -> int:
        return sum(weight for pc, weight in bar_notes if pc in triad)

    best = max((score(triad) for triad in triads.values()), default=0)
    return score(declared) >= best


@dataclass(frozen=True)
class ThemeCandidate:
    items: tuple[MelodyItem, ...]
    target: TargetProfile
    features: FeatureSummary
    conformance: float
    pin_report: tuple[str, ...]
    pool_index: int = -1     # position in the kept pool it was generated at (run-log trace)
    backtracks: int = 0      # backtracks this candidate cost to generate (run-log trace)
    soft_pin_fails: int = 0  # harmonic/cadence implication misses (guided, not dropped)


@dataclass(frozen=True)
class CandidateAttempt:
    """One Stage-1 generation attempt and why it did or didn't enter the pool (run-log)."""

    index: int                       # attempt number, 1-based
    outcome: str                     # "kept" | "abandoned" | "duplicate" | "pin_fail"
    backtracks: int = 0
    pool_index: int | None = None    # kept-pool position, if kept
    conformance: float | None = None
    soft_pin_fails: int = 0          # harmonic/cadence implication misses, if kept


@dataclass
class GenerationTrace:
    """A populated-by-the-engine record of one ``generate_theme_batch`` run (run-log).

    Pass an empty instance into ``generate_theme_batch``; it is filled in place. ``attempts``
    is the per-attempt outcome stream; ``kept_after_cut`` and ``selected`` are kept-pool indices
    surviving the conformance keep-cut and chosen into the final spread, respectively.
    """

    pool_target: int = 0
    batch_size: int = 0
    attempts: list[CandidateAttempt] = field(default_factory=list)
    kept_after_cut: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()


def _plan_echo_rate(similarity) -> float:
    """Per-bar quote probability for a candidate, from its rolled self-similarity plan (D7).

    The mean off-diagonal target similarity: an echo-dense plan (more bars meant to recur) asks
    the generator to quote the preceding bar more often. Capped by ``ECHO_RATE_CAP``.
    """
    rows = list(similarity)
    size = len(rows)
    if size < 2:
        return 0.0
    total = sum(rows[i][j] for i in range(size) for j in range(size) if i != j)
    return min(ECHO_RATE_GAIN * total / (size * (size - 1)), ECHO_RATE_CAP)


def generate_theme_batch(
    kernel: ThemeKernel,
    corpus: Iterable[Sequence[MelodyItem]] | None = None,
    *,
    model: MarkovMelodyModel | None = None,
    pool_size: int = 96,
    batch_size: int = 12,
    seed: int | None = None,
    weights: ConformanceWeights = ConformanceWeights(),
    keep_fraction: float = 0.6,
    trace: GenerationTrace | None = None,
) -> list[ThemeCandidate]:
    """Generate a dissimilar batch inside one kernel.

    The melodic statistics come from ``model`` if given (a pre-trained / loaded
    ``MarkovMelodyModel``), else from training on ``corpus``, else the repo themes. The
    returned order is the spread-selection order. It is not a quality ranking. If a
    ``GenerationTrace`` is passed, it is filled in place with the run's per-attempt outcomes
    (run-log).
    """

    if pool_size < batch_size:
        raise ValueError("pool_size must be >= batch_size")

    rng = Random(seed)
    if model is None:
        model = MarkovMelodyModel.from_corpus(corpus or default_theme_corpus())
    _validate_kernel(kernel)

    raw: list[ThemeCandidate] = []
    attempts: list[CandidateAttempt] = []
    seen: set[tuple[tuple[Any, float], ...]] = set()
    max_attempts = max(pool_size * 30, 200)
    for attempt_no in range(1, max_attempts + 1):
        if len(raw) >= pool_size:
            break
        # Roll this candidate's target first so its per-bar contour plan steers generation (D7),
        # then score the realized line against the same (per-candidate randomized) target.
        target = roll_target_profile(kernel, rng)
        echo_rate = _plan_echo_rate(target.similarity)
        stats = {"backtracks": 0}
        generated = _generate_constrained_items(kernel, model, rng, stats=stats,
                                                plan=target.contour, echo_rate=echo_rate)
        backtracks = stats["backtracks"]
        if generated is None:
            attempts.append(CandidateAttempt(attempt_no, "abandoned", backtracks))
            continue
        items = tuple(generated)
        key = tuple((_pitch_key(item[0]), _round_duration(float(item[1]))) for item in items)
        if key in seen:
            attempts.append(CandidateAttempt(attempt_no, "duplicate", backtracks))
            continue
        seen.add(key)
        hard_ok, soft_fails = pin_outcome(kernel, items)
        if not hard_ok:
            attempts.append(CandidateAttempt(attempt_no, "pin_fail", backtracks))
            continue
        report = pins_satisfied(kernel, items)
        pool_index = len(raw)
        features = measure_free_features(kernel, items, model)
        conformance = conformance_distance(features, target, weights) + SOFT_PIN_PENALTY * soft_fails
        raw.append(ThemeCandidate(items, target, features, conformance, tuple(report),
                                  pool_index, backtracks, soft_fails))
        attempts.append(CandidateAttempt(attempt_no, "kept", backtracks, pool_index,
                                         conformance, soft_fails))

    if len(raw) < batch_size:
        raise RuntimeError(f"only generated {len(raw)} candidates; requested {batch_size}")

    keep = max(batch_size, min(len(raw), int(len(raw) * keep_fraction)))
    conforming = sorted(raw, key=lambda candidate: candidate.conformance)[:keep]
    selected = _select_spread(conforming, batch_size)

    if trace is not None:
        trace.pool_target = pool_size
        trace.batch_size = batch_size
        trace.attempts = attempts
        trace.kept_after_cut = tuple(candidate.pool_index for candidate in conforming)
        trace.selected = tuple(candidate.pool_index for candidate in selected)
    return selected


def _pitch_rhythm_pin_ok(frame: ThemeFrame, by_offset: dict, pin) -> bool:
    """Whether a pitch/rhythm pin's exact notes land at its offsets (the hard identity check)."""
    offset = _pin_offset(frame, pin)
    for expected in pin.items:
        found = by_offset.get(_tick(offset))
        if found is None or not _same_pitch_duration(found, expected):
            return False
        offset += float(expected[1])
    return True


def pin_outcome(kernel: ThemeKernel, items: Sequence[MelodyItem]) -> tuple[bool, int]:
    """``(hard_ok, soft_fail_count)`` for a candidate.

    Hard = pitch/rhythm pins, the inviolable thematic identity (the head riff): a hard failure
    discards the candidate. Soft = the harmonic-implication and cadence-articulation checks,
    which are *guided toward* (downbeat chord tones + harmonic-context steering) but no longer
    a drop gate — they only penalize conformance, so the un-conforming majority still enters the
    pool and the spread keeps its variety instead of collapsing onto the conventional 15%.
    """
    timed = _time_items(items)
    by_offset = {_tick(t.offset): t.item for t in timed}
    hard_ok = all(_pitch_rhythm_pin_ok(kernel.frame, by_offset, pin)
                  for pin in kernel.pitch_rhythm_pins)
    soft = 0
    for pin in kernel.harmonic_pins:
        if pin.chord_tones and not _harmonic_implies(kernel.frame, pin, timed):
            soft += 1
    for pin in kernel.structural_pins:
        if not _cadence_articulated(kernel.frame, pin, timed):
            soft += 1
    return hard_ok, soft


def pins_satisfied(kernel: ThemeKernel, items: Sequence[MelodyItem]) -> tuple[str, ...]:
    timed = _time_items(items)
    by_offset = {_tick(t.offset): t.item for t in timed}
    report: list[str] = []

    for pin in sorted(kernel.pitch_rhythm_pins, key=lambda p: _pin_offset(kernel.frame, p)):
        ok = _pitch_rhythm_pin_ok(kernel.frame, by_offset, pin)
        report.append(f"{pin.name}: {'OK' if ok else 'FAIL'}")

    for pin in kernel.harmonic_pins:
        if not pin.chord_tones:
            report.append(f"harmonic b{pin.bar} {pin.label}: OK")
            continue
        ok = _harmonic_implies(kernel.frame, pin, timed)
        report.append(f"harmonic b{pin.bar} {pin.label}: {'OK' if ok else 'FAIL'}")

    for pin in kernel.structural_pins:
        ok = _cadence_articulated(kernel.frame, pin, timed)
        report.append(f"structural b{pin.start_bar}-{pin.end_bar} {pin.label}: {'OK' if ok else 'FAIL'}")

    return tuple(report)


def _active_chord(kernel: ThemeKernel, offset: float) -> tuple[int, str] | None:
    """The ``(root_pc, quality)`` of the harmonic pin governing ``offset``, or ``None`` (D6).

    The chord-relative harmonic context the generator conditions on: the latest declared
    harmonic pin at or before this bar, with quality read from its chord tones (major if a
    major 3rd above the root is present, else minor). Mirrors the chord inference used in
    training, so the learned harmonic function transfers to the kernel's declared chords.
    """
    bar = int(offset // kernel.frame.beats_per_bar) + 1
    governing = None
    for pin in kernel.harmonic_pins:
        if pin.chord_tones and pin.bar <= bar and (governing is None or pin.bar > governing.bar):
            governing = pin
    if governing is None:
        return None
    pcs = _pitch_classes(governing.chord_tones)
    root = next(iter(_pitch_classes((governing.chord_tones[0],))))
    quality = "maj" if (root + 4) % 12 in pcs else "min" if (root + 3) % 12 in pcs else "maj"
    return (root, quality)


def _legal_pitches(kernel: ThemeKernel, offset: float = 0.0) -> list[int]:
    """Every chromatic pitch within the frame's range — the model chooses among them.

    NO scale filtering and NO chord-tone-on-downbeat filtering: discarding the corpus's
    non-scale content (chromatic passing tones, blue notes, borrowed colour) was throwing away
    exactly the material that makes a line sound learned rather than quantised to a scale. Range
    is the only hard pitch bound; tonal/harmonic pull comes from the learned interval + harmonic
    context and the soft pins, not a filter. ``offset`` is unused (kept for call-site stability).
    """
    low, high = _frame_bounds(kernel.frame)
    return list(range(low, high + 1))


def _palette_ticks(frame: ThemeFrame) -> tuple[int, ...]:
    """The frame's duration palette as sorted unique tick lengths."""
    return tuple(sorted({_duration_to_ticks(duration) for duration in _duration_palette(frame)}))


def _lands(pitch: int | None, next_pin_pitch: int | None, cadence_pc: int | None = None) -> bool:
    """Whether ``pitch`` is a clean landing: the cadence pitch class if one is required
    (M4 frame-final cadence), else within ``MAX_LANDING_LEAP`` of the next pin."""
    if cadence_pc is not None:
        return pitch is not None and pitch % 12 == cadence_pc
    return next_pin_pitch is None or (
        pitch is not None and abs(next_pin_pitch - pitch) <= MAX_LANDING_LEAP
    )


def _segment_feasibility(
    kernel: ThemeKernel,
    start: float,
    end: float,
    next_pin_pitch: int | None,
    cadence_pc: int | None = None,
):
    """A memoized feasibility predicate ``feasible(remaining_ticks, prev_pitch)`` for one
    segment ``[start, end)`` (D2).

    Proves, before any move is committed, that the span can be filled exactly under
    scale/range/palette and the harmonic constraint, ending within a clean leap of the pin
    (or, for a frame-final cadential segment, on the cadence pitch class — M4).
    The closure shares one memo so the sampler can query every candidate move cheaply.
    Memoized on ``(remaining_ticks, prev_pitch)`` (offset is implied by remaining_ticks).
    """
    end_ticks = _duration_to_ticks(end)
    palette = _palette_ticks(kernel.frame)
    memo: dict[tuple[int, int | None], bool] = {}

    def feasible(remaining: int, pitch: int | None) -> bool:
        if remaining == 0:
            return _lands(pitch, next_pin_pitch, cadence_pc)
        key = (remaining, pitch)
        if key in memo:
            return memo[key]
        offset = (end_ticks - remaining) / TICKS_PER_QUARTER
        legal = _legal_pitches(kernel, offset)
        result = False
        for duration_ticks in palette:
            if duration_ticks > remaining:
                continue
            for candidate in legal:
                if feasible(remaining - duration_ticks, candidate):
                    result = True
                    break
            if result:
                break
        memo[key] = result
        return result

    return feasible


def _segment_feasible(
    kernel: ThemeKernel,
    start: float,
    remaining_ticks: int,
    prev_pitch: int | None,
    next_pin_pitch: int | None,
) -> bool:
    """Whether some legal completion of the segment exists from this state (D2)."""
    end = start + remaining_ticks / TICKS_PER_QUARTER
    return _segment_feasibility(kernel, start, end, next_pin_pitch)(remaining_ticks, prev_pitch)


def _cadence_stops(kernel: ThemeKernel, pins) -> list[tuple[float, int]]:
    """Free-space cadence landing boundaries: (span-end offset, target pc) for each
    cadential structural span whose end is not already covered by a pin (M4).

    A cadence that lands on a pin is enforced by the pin material, not by generation.
    """
    frame = kernel.frame
    pin_spans = [(off, off + sum(float(item[1]) for item in items)) for off, items in pins]
    stops: list[tuple[float, int]] = []
    for pin in kernel.structural_pins:
        target = _cadence_target_pc(frame, pin.cadence)
        if target is None:
            continue
        offset = pin.end_bar * frame.beats_per_bar
        if any(start - EPS <= offset <= end + EPS for start, end in pin_spans):
            continue
        stops.append((offset, target))
    return stops


def _kernel_stops(kernel: ThemeKernel) -> list[tuple[float, int, object]]:
    """Ordered generation stops: ``(offset, kind, payload)`` where kind 0 = pitch/rhythm
    pin (payload = items), kind 1 = free-space cadence boundary (payload = target pc).

    A pin sorts before a cadence at the same offset. Single source of truth for the
    segment boundaries shared by generation and density reporting (D5).
    """
    pins = _pin_timed_items(kernel)
    stops: list[tuple[float, int, object]] = [(off, 0, items) for off, items in pins]
    stops += [(off, 1, target) for off, target in _cadence_stops(kernel, pins)]
    stops.sort(key=lambda stop: (stop[0], stop[1]))
    return stops


def _free_segments(kernel: ThemeKernel) -> list[tuple[float, float, int | None, int | None]]:
    """The fill gaps between stops as ``(start, end, next_pin_pitch, cadence_pc)``, in
    order, including the final fill to the frame end (D5). Mirrors the generation walk."""
    frame = kernel.frame
    segments: list[tuple[float, float, int | None, int | None]] = []
    current = 0.0
    for offset, kind, payload in _kernel_stops(kernel):
        if offset - current > EPS:
            if kind == 0:
                segments.append((current, offset, _first_pitch(payload), None))
            else:
                segments.append((current, offset, None, payload))
        if kind == 0:
            current = offset + sum(float(item[1]) for item in payload)
        else:
            current = offset
    if frame.total_ql - current > EPS:
        segments.append((current, frame.total_ql, None, None))
    return segments


def _generate_constrained_items(
    kernel: ThemeKernel,
    model: MarkovMelodyModel,
    rng: Random,
    stats: dict | None = None,
    plan: tuple[float, ...] | None = None,
    echo_rate: float = 0.0,
) -> list[MelodyItem] | None:
    """Build one candidate, or ``None`` if any segment is abandoned (D2).

    Generation is driven by ordered stops: pitch/rhythm pins (place material, land by leap)
    and free-space cadence boundaries (land the segment's final note on the cadence pitch
    class, M4). Segments fill the gaps between consecutive stops. ``stats``, if given,
    accumulates this candidate's total backtracks under ``"backtracks"`` (run-log trace).
    """
    frame = kernel.frame
    pins = _pin_timed_items(kernel)
    low, high = _frame_bounds(frame)

    stops = _kernel_stops(kernel)

    current = 0.0
    out: list[MelodyItem] = []
    previous_pitch = _initial_pitch(frame, pins)
    previous_interval = 0
    previous_duration = None

    def fill_to(offset, next_pin_pitch, cadence_pc):
        nonlocal previous_pitch, previous_interval, previous_duration
        if offset - current <= EPS:
            return True
        start_pitch = previous_pitch if previous_pitch is not None else (low + high) // 2
        generated = _generate_segment(
            kernel, model, rng, current, offset,
            start_pitch, previous_interval, previous_duration, next_pin_pitch,
            cadence_pc=cadence_pc, stats=stats, plan=plan, echo_rate=echo_rate,
        )
        if generated is None:
            return False
        segment, previous_pitch, previous_interval, previous_duration = generated
        out.extend(segment)
        return True

    for offset, kind, payload in stops:
        if kind == 0:  # pitch/rhythm pin
            if not fill_to(offset, _first_pitch(payload), None):
                return None
            for item in payload:
                normalized = _normalize_item(item)
                out.append(normalized)
                midi = _item_midi(normalized)
                if midi is not None:
                    previous_interval = midi - previous_pitch if previous_pitch is not None else 0
                    previous_pitch = midi
                previous_duration = _round_duration(float(normalized[1]))
            current = offset + sum(float(item[1]) for item in payload)
        else:  # cadence boundary: the segment's final note is the cadence note
            if not fill_to(offset, None, payload):
                return None
            current = offset

    if frame.total_ql - current > EPS:
        if not fill_to(frame.total_ql, None, None):
            return None
        current = frame.total_ql

    total = sum(float(item[1]) for item in out)
    if abs(total - frame.total_ql) > EPS:
        raise RuntimeError(f"candidate has {total} ql, expected {frame.total_ql}")
    return out


def _pitch_weight(
    frame: ThemeFrame,
    model: MarkovMelodyModel,
    previous_pitch: int | None,
    previous_interval: int,
    next_pin_pitch: int | None,
    midi: int,
    context: Context | None = None,
) -> float:
    """The learned interval weight for placing ``midi``, plus only the pin-landing bias.

    The melodic shaping is the Markov chain's alone: NO hand-coded leap or repeated-note
    penalties (the corpus already encodes how often those occur — re-penalising them was just
    overriding the data with taste). ``context`` (D3) conditions the interval probability on
    meter accent, interval history, and harmonic function. The one non-corpus factor is the bias
    toward landing near the next pin, which the corpus cannot know about (pin steering).
    """
    low, high = _frame_bounds(frame)
    center = (low + high) / 2.0
    interval = int(round(midi - center)) if previous_pitch is None else midi - previous_pitch
    weight = model.interval_weight(previous_interval, interval, context=context)
    if next_pin_pitch is not None:
        weight *= 0.65 + exp(-abs(next_pin_pitch - midi) / 5.0)
    return weight


def _weighted_order(moves, weights, rng):
    """Yield moves in weighted-random order, drawing lazily (one rng draw per yield).

    The first draw matches ``_weighted_choice``, so the happy path (no backtrack) is
    identical to a single forward weighted sample.
    """
    moves = list(moves)
    weights = [max(weight, 0.0) for weight in weights]
    while moves:
        total = sum(weights)
        if total <= EPS:
            index = rng.randrange(len(moves))
        else:
            roll = rng.random() * total
            accumulator = 0.0
            index = len(moves) - 1
            for i, weight in enumerate(weights):
                accumulator += weight
                if accumulator >= roll:
                    index = i
                    break
        yield moves.pop(index)
        weights.pop(index)


def _echo_matches(midi: int | None, interval: int | None, prev_pitch: int | None) -> bool:
    """Whether a candidate move reproduces a quoted cell element (rest, or transposed note)."""
    if interval is None:
        return midi is None
    return midi is not None and prev_pitch is not None and midi - prev_pitch == interval


def _relative_bar_cell(acc, beats_per_bar) -> list[tuple[int | None, float]] | None:
    """The immediately preceding whole bar of ``acc`` as ``(interval_from_prev | None, dur)``.

    A transposable motif cell: each sounding note carries the interval that led into it (so the
    replay tracks the line's current pitch), rests carry ``None``. ``None`` if the tail of
    ``acc`` does not align to exactly one bar (a ragged or too-short segment — no clean motif).
    """
    bar_ticks = _duration_to_ticks(beats_per_bar)
    acc_ticks = 0
    start_index = len(acc)
    for index in range(len(acc) - 1, -1, -1):
        acc_ticks += _duration_to_ticks(acc[index][1])
        start_index = index
        if acc_ticks >= bar_ticks:
            break
    if acc_ticks != bar_ticks:
        return None
    running = None
    for index in range(start_index - 1, -1, -1):
        if acc[index][0] is not None:
            running = _item_midi(acc[index])
            break
    cell: list[tuple[int | None, float]] = []
    for index in range(start_index, len(acc)):
        pitch, duration = acc[index][0], acc[index][1]
        if pitch is None:
            cell.append((None, duration))
        else:
            midi = _item_midi(acc[index])
            cell.append(((midi - running) if running is not None else 0, duration))
            running = midi
    return cell


def _search_segment(
    kernel: ThemeKernel,
    model: MarkovMelodyModel,
    rng: Random,
    start: float,
    end: float,
    previous_pitch: int | None,
    previous_interval: int,
    previous_duration: float | None,
    next_pin_pitch: int | None,
    cadence_pc: int | None = None,
    plan: tuple[float, ...] | None = None,
    echo_rate: float = 0.0,
) -> tuple[tuple[list[MelodyItem], int | None, int, float | None] | None, int]:
    """Feasibility-gated depth-first fill of ``[start, end)`` with bounded backtracking (D2).

    Returns ``(result, backtracks)`` where ``result`` is
    ``(items, prev_pitch, prev_interval, prev_duration)`` or ``None`` if the segment is
    abandoned. Only feasible moves are tried, so from a feasible state the first move
    succeeds and no backtracking occurs; backtracking is the safety net for over-tight
    regions, bounded by ``MAX_BACKTRACKS`` (swappable) before the candidate is abandoned.
    ``cadence_pc`` (M4) forces the final note onto a cadence pitch class.
    """
    frame = kernel.frame
    beats_per_bar = frame.beats_per_bar
    order = model.order
    low, high = _frame_bounds(frame)
    span = max(high - low, 1)
    feasible = _segment_feasibility(kernel, start, end, next_pin_pitch, cadence_pc)
    palette = _palette_ticks(frame)
    total_ticks = _duration_to_ticks(end - start)
    state = {"backtracks": 0}

    def steer(offset, midi):
        """Soft pull of ``midi`` toward the candidate's planned register for this bar (D7)."""
        if not plan:
            return 1.0
        bar = int(offset // beats_per_bar)
        if not 0 <= bar < len(plan):
            return 1.0
        planned = low + plan[bar] * span
        return exp(-STEER_STRENGTH * abs(midi - planned) / span)

    def step_context(offset, prev_pitch, history, after_rest_or_start):
        """The D3 conditioning context for the note about to be placed at ``offset``.

        ``accent``/``harmonic``/``history`` are exact; ``phrase`` is the generation-side proxy
        for training's rest-delimited phrase position: S at a segment start or just after a
        rest, M otherwise (E notes are cadence-/pin-forced, so they need no interval steering).
        """
        chord = _active_chord(kernel, offset)
        return Context(
            history=tuple(history[-order:]),
            accent=_accent_bucket(offset, beats_per_bar),
            phrase="S" if after_rest_or_start else "M",
            harmonic=_harmonic_function_bucket(
                prev_pitch % 12 if prev_pitch is not None else None, chord
            ),
        )

    def search(remaining, offset, prev_pitch, prev_interval, prev_duration, history,
               acc, echo_cell=None, echo_pos=0):
        if remaining == 0:
            return (list(acc), prev_pitch, prev_interval, prev_duration)
        after_rest_or_start = (not acc) or acc[-1][0] is None
        ctx = step_context(offset, prev_pitch, history, after_rest_or_start)

        # Motif reuse (off unless ECHO_ENABLED): begin quoting the preceding bar at a bar-start.
        if (ECHO_ENABLED and echo_cell is None and echo_rate > 0.0
                and abs(offset % beats_per_bar) < EPS and acc
                and remaining >= _duration_to_ticks(beats_per_bar)
                and rng.random() < echo_rate):
            echo_cell = _relative_bar_cell(acc, beats_per_bar)
            echo_pos = 0

        legal = _legal_pitches(kernel)  # full chromatic range; the model chooses
        moves: list[tuple[float, int | None]] = []
        weights: list[float] = []
        for duration_ticks in palette:
            if duration_ticks > remaining:
                continue
            duration = duration_ticks / TICKS_PER_QUARTER
            for midi in legal:
                if feasible(remaining - duration_ticks, midi):
                    ibucket = _interval_bucket(midi - prev_pitch) if prev_pitch is not None else "_"
                    moves.append((duration, midi))
                    weights.append(
                        model.duration_weight(prev_duration, duration, context=ctx,
                                              interval_bucket=ibucket)
                        * _pitch_weight(frame, model, prev_pitch, prev_interval,
                                        next_pin_pitch, midi, context=ctx)
                        * steer(offset, midi)
                    )

        note_mass = sum(weights)

        # Rests come from the model: P(rest | accent) learned from the corpus sets the rest mass
        # relative to the note mass, split across feasible rest durations by the learned
        # rest-duration profile. The only constraint kept is feasibility (a rest can't be the
        # landing note) so the segment still reaches its pin — no hand-set rate or placement rules.
        rest_p = model.rest_probability(ctx)
        if note_mass > EPS and rest_p > 0.0:
            rest_moves: list[tuple[float, None]] = []
            rest_base: list[float] = []
            for duration_ticks in palette:
                if duration_ticks >= remaining:
                    continue
                rest_ql = duration_ticks / TICKS_PER_QUARTER
                if feasible(remaining - duration_ticks, prev_pitch):
                    rest_moves.append((rest_ql, None))
                    rest_base.append(model.rest_duration_weight(rest_ql))
            base_sum = sum(rest_base)
            if rest_moves and base_sum > EPS:
                rest_scale = (rest_p / (1.0 - rest_p)) * note_mass / base_sum
                for move, base in zip(rest_moves, rest_base):
                    moves.append(move)
                    weights.append(base * rest_scale)

        # Strongly prefer the next quoted move while an echo is active and the quoted note is a
        # feasible option (else the quote breaks and free generation resumes).
        echo_active = echo_cell is not None and echo_pos < len(echo_cell)
        if echo_active:
            quote_interval, quote_duration = echo_cell[echo_pos]
            for index, (move_duration, move_midi) in enumerate(moves):
                if abs(move_duration - quote_duration) < EPS and _echo_matches(
                    move_midi, quote_interval, prev_pitch
                ):
                    weights[index] *= ECHO_BIAS

        for duration, midi in _weighted_order(moves, weights, rng):
            next_ticks = remaining - _duration_to_ticks(duration)
            next_echo_cell, next_echo_pos = None, 0
            if echo_active and _echo_matches(midi, echo_cell[echo_pos][0], prev_pitch) \
                    and abs(duration - echo_cell[echo_pos][1]) < EPS \
                    and echo_pos + 1 < len(echo_cell):
                next_echo_cell, next_echo_pos = echo_cell, echo_pos + 1
            if midi is None:
                acc.append((None, duration))
                result = search(next_ticks, offset + duration, prev_pitch, prev_interval,
                                _round_duration(duration), history, acc,
                                next_echo_cell, next_echo_pos)
            else:
                acc.append((_spell_midi(frame, midi), duration))
                next_interval = midi - prev_pitch if prev_pitch is not None else prev_interval
                next_history = history + (next_interval,) if prev_pitch is not None else history
                result = search(next_ticks, offset + duration, midi, next_interval,
                                _round_duration(duration), next_history, acc,
                                next_echo_cell, next_echo_pos)
            if result is not None:
                return result
            acc.pop()
            state["backtracks"] += 1
            if state["backtracks"] > MAX_BACKTRACKS:
                return None
        return None

    result = search(total_ticks, start, previous_pitch, previous_interval, previous_duration,
                    (), [])
    return result, state["backtracks"]


def _generate_segment(
    kernel: ThemeKernel,
    model: MarkovMelodyModel,
    rng: Random,
    start: float,
    end: float,
    previous_pitch: int | None,
    previous_interval: int,
    previous_duration: float | None,
    next_pin_pitch: int | None,
    cadence_pc: int | None = None,
    stats: dict | None = None,
    plan: tuple[float, ...] | None = None,
    echo_rate: float = 0.0,
) -> tuple[list[MelodyItem], int | None, int, float | None] | None:
    """Fill ``[start, end)`` with the feasibility-gated bounded-backtracking search (D2).

    Returns the segment 4-tuple, or ``None`` if the segment is abandoned (over-tight).
    ``cadence_pc`` (M4) forces the final note onto a cadence pitch class. ``stats``, if given,
    accumulates the segment's backtrack count under ``"backtracks"`` (run-log trace). ``plan``
    (D7) is the candidate's per-bar contour that softly steers free pitch choices; ``echo_rate``
    (D7) is its per-bar chance of quoting the preceding bar (motif reuse).
    """
    result, backtracks = _search_segment(
        kernel, model, rng, start, end,
        previous_pitch, previous_interval, previous_duration, next_pin_pitch,
        cadence_pc=cadence_pc, plan=plan, echo_rate=echo_rate,
    )
    if stats is not None:
        stats["backtracks"] = stats.get("backtracks", 0) + backtracks
    return result


def _select_spread(candidates: Sequence[ThemeCandidate], batch_size: int) -> list[ThemeCandidate]:
    if len(candidates) <= batch_size:
        return list(candidates)
    vectors = [_feature_vector(candidate.features) for candidate in candidates]
    first = max(
        range(len(candidates)),
        key=lambda i: (
            sum(_vector_distance(vectors[i], vectors[j]) for j in range(len(candidates)) if j != i)
            / max(len(candidates) - 1, 1),
            -candidates[i].conformance,
        ),
    )
    selected_indexes = [first]
    remaining = [i for i in range(len(candidates)) if i != first]
    while remaining and len(selected_indexes) < batch_size:
        chosen = max(
            remaining,
            key=lambda i: (
                min(_vector_distance(vectors[i], vectors[s]) for s in selected_indexes),
                -candidates[i].conformance,
            ),
        )
        selected_indexes.append(chosen)
        remaining.remove(chosen)
    return [candidates[i] for i in selected_indexes]
