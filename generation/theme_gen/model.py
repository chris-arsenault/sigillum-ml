"""The Markov melody model (the Stage-1 generator / Stage-2 surprise meter).

A small first-order model over interval and duration transitions, trained from a corpus of
item-lists. Decision D3 (``docs/architecture/17_theme_generator.md``) deepens this into a
context-conditioned, persisted model (plan M2). This module carries the original
prototype's Markov model forward (plan M0).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import exp, log
from pathlib import Path
from typing import Iterable, Sequence

from generation.theme_gen._common import (
    EPS,
    MelodyItem,
    _counter_probability,
    _item_midi,
    _round_duration,
)


def _accent_bucket(offset: float, beats_per_bar: float) -> str:
    """Metric-accent bucket of an offset under an assumed meter (D3 meter accent).

    D = downbeat, H = half-bar, B = other on-beat (integer ql), O = off-beat.
    """
    position = offset % beats_per_bar
    if abs(position) < EPS or abs(position - beats_per_bar) < EPS:
        return "D"
    if abs(position - beats_per_bar / 2.0) < EPS:
        return "H"
    if abs(position - round(position)) < EPS:
        return "B"
    return "O"


def _phrase_bucket(index: int, length: int) -> str:
    """Phrase-position bucket within a rest-delimited phrase (D3 phrase position).

    S = phrase start, E = phrase end (cadential), M = medial.
    """
    if index <= 0:
        return "S"
    if index >= length - 1:
        return "E"
    return "M"


def _interval_bucket(interval: int | None) -> str:
    """Magnitude class of the interval taken *into* a note (rhythm<->pitch coupling, D8).

    rep = repeat, step = 1-2 semitones, leap = 3-7, wide = octave+. Direction is dropped:
    what couples to duration is how far the line jumped, not which way.
    """
    if interval is None:
        return "_"
    magnitude = abs(interval)
    if magnitude == 0:
        return "rep"
    if magnitude <= 2:
        return "step"
    if magnitude <= 7:
        return "leap"
    return "wide"


# Triads as pitch-class offsets from the root (the harmonic vocabulary chord inference fits).
_TRIAD_QUALITIES: dict[str, tuple[int, ...]] = {"maj": (0, 4, 7), "min": (0, 3, 7)}


def _best_triad(pc_weights) -> tuple[int, str] | None:
    """Best-fit ``(root_pc, quality)`` for a pitch-class -> duration-weight histogram (D6).

    Picks the major/minor triad covering the most weighted pitch-class mass; ties break to the
    lowest root then major. ``None`` for an empty window. A cheap stand-in for real harmony on a
    bare melody — noisy per bar, but the aggregate captures chord-tone stability and resolution.
    """
    if not pc_weights:
        return None
    best: tuple[int, str] | None = None
    best_score = -1.0
    for root in range(12):
        for quality, offsets in _TRIAD_QUALITIES.items():
            tones = {(root + off) % 12 for off in offsets}
            score = sum(weight for pc, weight in pc_weights.items() if pc in tones)
            if score > best_score:
                best_score = score
                best = (root, quality)
    return best


def _bar_chords(notes, beats_per_bar) -> dict[int, tuple[int, str] | None]:
    """Infer one chord per bar from ``(offset, midi, duration)`` sounding notes (D6)."""
    histograms: dict[int, Counter] = {}
    for offset, midi, duration in notes:
        bar = int(offset // beats_per_bar)
        histograms.setdefault(bar, Counter())[midi % 12] += duration
    return {bar: _best_triad(histogram) for bar, histogram in histograms.items()}


def _harmonic_function_bucket(note_pc: int | None, chord: tuple[int, str] | None) -> str:
    """Key-invariant harmonic context: chord quality + chromatic degree of the note above the
    chord root (D6). ``min0`` = root of a minor chord, ``maj4`` = 3rd of a major chord, etc.

    Key-invariant by construction, so a function learned from corpus melodies in any key
    transfers to a kernel's declared chords. ``"_"`` when the chord or note is unknown.
    """
    if chord is None or note_pc is None:
        return "_"
    root, quality = chord
    return f"{quality}{(note_pc - root) % 12}"


# D3 defaults (M2 [DECISION]): order bound, backoff min-count, assumed meter.
DEFAULT_ORDER = 2
DEFAULT_MIN_COUNT = 20
DEFAULT_BEATS_PER_BAR = 4.0


@dataclass(frozen=True)
class Context:
    """The conditioning context for one predicted event (D3).

    ``history`` is the last ``order`` intervals; ``accent``/``phrase``/``harmonic`` are the
    buckets of the event being predicted. Defaults reproduce a phrase-start downbeat.
    """

    history: tuple[int, ...] = ()
    accent: str = "D"
    phrase: str = "S"
    harmonic: str = "_"


def _interval_backoff_keys(history, accent: str, phrase: str, harmonic: str) -> list[str]:
    """Interval context keys, most-specific to least (stupid-backoff chain).

    Covers the richer levels above the first-order ``interval_counts`` backbone:
    history+accent+phrase+harmonic -> +accent+phrase -> +accent -> history-only.
    """
    h = ",".join(str(i) for i in history)
    return [
        f"h={h}|a={accent}|p={phrase}|m={harmonic}",
        f"h={h}|a={accent}|p={phrase}",
        f"h={h}|a={accent}",
        f"h={h}",
    ]


def _duration_backoff_keys(
    prev_duration, accent: str, phrase: str, interval_bucket: str | None = None
) -> list[str]:
    """Duration context keys, most-specific to least, above the first-order backbone.

    When ``interval_bucket`` is given (the magnitude of the interval taken into this note), the
    rhythm<->pitch coupled keys lead the chain (D8), backing off to the uncoupled keys — so a
    leap can carry its own duration distribution where the data supports it, else fall through.
    """
    d = _round_duration(prev_duration) if prev_duration is not None else None
    keys: list[str] = []
    if interval_bucket is not None and interval_bucket != "_":
        keys.append(f"d={d}|a={accent}|p={phrase}|i={interval_bucket}")
        keys.append(f"d={d}|a={accent}|i={interval_bucket}")
    keys.append(f"d={d}|a={accent}|p={phrase}")
    keys.append(f"d={d}|a={accent}")
    return keys


def _accumulate_context(line, order, beats_per_bar, interval_ctx, duration_ctx):
    """Increment hierarchical context counts for one line (rest-delimited phrases)."""
    offset = 0.0
    phrases: list[list[tuple[float, int, float]]] = []
    current: list[tuple[float, int, float]] = []
    for item in line:
        duration = _round_duration(float(item[1]))
        midi = _item_midi(item)
        if midi is None:
            if current:
                phrases.append(current)
                current = []
        else:
            current.append((offset, midi, duration))
        offset += duration
    if current:
        phrases.append(current)

    chords = _bar_chords((note for phrase in phrases for note in phrase), beats_per_bar)

    for phrase in phrases:
        length = len(phrase)
        history: list[int] = []
        prev_pitch = None
        prev_duration = None
        for index, (off, midi, duration) in enumerate(phrase):
            accent = _accent_bucket(off, beats_per_bar)
            phr = _phrase_bucket(index, length)
            chord = chords.get(int(off // beats_per_bar))
            harmonic = _harmonic_function_bucket(
                prev_pitch % 12 if prev_pitch is not None else None, chord
            )
            if prev_pitch is not None:
                interval = midi - prev_pitch
                ibucket = _interval_bucket(interval)
                hist = tuple(history[-order:])
                for key in _interval_backoff_keys(hist, accent, phr, harmonic):
                    interval_ctx.setdefault(key, Counter())[interval] += 1
                history.append(interval)
            else:
                ibucket = "_"
            if prev_duration is not None:
                for key in _duration_backoff_keys(prev_duration, accent, phr, ibucket):
                    duration_ctx.setdefault(key, Counter())[duration] += 1
            prev_pitch = midi
            prev_duration = duration


class MarkovMelodyModel:
    """Small first-order melodic meter/generator over Sigillum item lists."""

    def __init__(
        self,
        interval_counts: dict[int, Counter[int]],
        duration_counts: dict[float, Counter[float]],
        global_intervals: Counter[int],
        global_durations: Counter[float],
        context_interval_counts: dict[str, Counter[int]] | None = None,
        context_duration_counts: dict[str, Counter[float]] | None = None,
        rest_counts: dict[str, int] | None = None,
        event_counts: dict[str, int] | None = None,
        global_rest: int = 0,
        global_events: int = 0,
        rest_duration_counts: Counter[float] | None = None,
        order: int = DEFAULT_ORDER,
        beats_per_bar: float = DEFAULT_BEATS_PER_BAR,
        min_count: int = DEFAULT_MIN_COUNT,
    ):
        self.interval_counts = interval_counts
        self.duration_counts = duration_counts
        self.global_intervals = global_intervals or Counter({0: 1, 2: 1, -2: 1})
        self.global_durations = global_durations or Counter({1.0: 1, 0.5: 1})
        self.context_interval_counts = context_interval_counts or {}
        self.context_duration_counts = context_duration_counts or {}
        # Rests as learned events (not a hand-set rate): how often, per metric-accent bucket, the
        # corpus's next event is a rest, plus the rest-duration distribution.
        self.rest_counts = rest_counts or {}
        self.event_counts = event_counts or {}
        self.global_rest = global_rest
        self.global_events = global_events
        self.rest_duration_counts = rest_duration_counts or Counter()
        self.order = order
        self.beats_per_bar = beats_per_bar
        self.min_count = min_count

    @classmethod
    def from_corpus(
        cls,
        corpus: Iterable[Sequence[MelodyItem]],
        *,
        order: int = DEFAULT_ORDER,
        beats_per_bar: float = DEFAULT_BEATS_PER_BAR,
        min_count: int = DEFAULT_MIN_COUNT,
    ) -> "MarkovMelodyModel":
        interval_counts: dict[int, Counter[int]] = defaultdict(Counter)
        duration_counts: dict[float, Counter[float]] = defaultdict(Counter)
        global_intervals: Counter[int] = Counter()
        global_durations: Counter[float] = Counter()
        context_interval_counts: dict[str, Counter[int]] = {}
        context_duration_counts: dict[str, Counter[float]] = {}
        rest_counts: dict[str, int] = defaultdict(int)
        event_counts: dict[str, int] = defaultdict(int)
        rest_duration_counts: Counter[float] = Counter()
        global_rest = 0
        global_events = 0

        for line in corpus:
            # First-order backbone (unchanged for back-compat).
            last_pitch = None
            last_interval = 0
            last_duration = None
            offset = 0.0
            for item in line:
                duration = _round_duration(float(item[1]))
                global_durations[duration] += 1
                if last_duration is not None:
                    duration_counts[last_duration][duration] += 1
                last_duration = duration

                midi = _item_midi(item)
                # Rest-as-event statistics, conditioned on metric accent (learned, not hand-set).
                accent = _accent_bucket(offset, beats_per_bar)
                event_counts[accent] += 1
                global_events += 1
                if midi is None:
                    rest_counts[accent] += 1
                    global_rest += 1
                    rest_duration_counts[duration] += 1
                offset += duration

                if midi is None:
                    continue
                if last_pitch is not None:
                    interval = midi - last_pitch
                    interval_counts[last_interval][interval] += 1
                    global_intervals[interval] += 1
                    last_interval = interval
                last_pitch = midi

            # Context-conditioned tables (D3): phrase position, meter accent, harmonic
            # proxy, variable order — accumulated over rest-delimited phrases.
            _accumulate_context(
                line, order, beats_per_bar, context_interval_counts, context_duration_counts
            )

        return cls(
            dict(interval_counts),
            dict(duration_counts),
            global_intervals,
            global_durations,
            context_interval_counts,
            context_duration_counts,
            dict(rest_counts),
            dict(event_counts),
            global_rest,
            global_events,
            rest_duration_counts,
            order=order,
            beats_per_bar=beats_per_bar,
            min_count=min_count,
        )

    def _backoff_probability(self, table, keys, value):
        """First context level (most specific) with total >= min_count, else None."""
        for key in keys:
            counter = table.get(key)
            if counter is not None and sum(counter.values()) >= self.min_count:
                return _counter_probability(counter, value)
        return None

    def duration_weight(
        self,
        previous: float | None,
        duration: float,
        context: "Context | None" = None,
        interval_bucket: str | None = None,
    ) -> float:
        duration = _round_duration(duration)
        if context is not None:
            keys = _duration_backoff_keys(
                previous, context.accent, context.phrase, interval_bucket
            )
            probability = self._backoff_probability(self.context_duration_counts, keys, duration)
            if probability is not None:
                return probability
        if previous is not None and previous in self.duration_counts:
            return _counter_probability(self.duration_counts[previous], duration)
        return _counter_probability(self.global_durations, duration)

    def interval_weight(
        self, previous_interval: int, interval: int, context: "Context | None" = None
    ) -> float:
        if context is not None:
            keys = _interval_backoff_keys(
                context.history, context.accent, context.phrase, context.harmonic
            )
            probability = self._backoff_probability(self.context_interval_counts, keys, interval)
            if probability is not None:
                return probability
        if previous_interval in self.interval_counts:
            return _counter_probability(self.interval_counts[previous_interval], interval)
        return _counter_probability(self.global_intervals, interval)

    def rest_probability(self, context: "Context | None" = None) -> float:
        """Learned P(next event is a rest), conditioned on metric accent when a context is given.

        Backs off to the global rest fraction on a sparse accent bucket. Returns 0 when the
        corpus had no rests (so an un-rebuilt model simply produces none) — never a hand-set rate.
        """
        if context is not None:
            total = self.event_counts.get(context.accent, 0)
            if total >= self.min_count:
                return self.rest_counts.get(context.accent, 0) / total
        return self.global_rest / self.global_events if self.global_events else 0.0

    def rest_duration_weight(self, duration: float) -> float:
        """Learned probability of a rest having ``duration`` (the corpus rest-length profile)."""
        return _counter_probability(self.rest_duration_counts, _round_duration(duration))

    def transition_surprise(
        self,
        previous_pitch: int | None,
        previous_interval: int,
        previous_duration: float | None,
        item: MelodyItem,
        context: "Context | None" = None,
    ) -> float:
        midi = _item_midi(item)
        duration = _round_duration(float(item[1]))
        if midi is not None and previous_pitch is not None:
            interval_bucket = _interval_bucket(midi - previous_pitch)
        else:
            interval_bucket = "_"
        d_prob = self.duration_weight(
            previous_duration, duration, context=context, interval_bucket=interval_bucket
        )
        if midi is None or previous_pitch is None:
            i_prob = _counter_probability(self.global_intervals, 0)
        else:
            i_prob = self.interval_weight(previous_interval, midi - previous_pitch, context=context)
        return -log(max(i_prob * d_prob, 1e-9))

    def _line_contexts(self, line) -> list["Context | None"]:
        """Per-item Context for sounding notes (None for rests), as training computes it."""
        contexts: list[Context | None] = [None] * len(line)
        offset = 0.0
        phrases: list[list[tuple[int, float, int, float]]] = []
        current: list[tuple[int, float, int, float]] = []
        for index, item in enumerate(line):
            duration = _round_duration(float(item[1]))
            midi = _item_midi(item)
            if midi is None:
                if current:
                    phrases.append(current)
                    current = []
            else:
                current.append((index, offset, midi, duration))
            offset += duration
        if current:
            phrases.append(current)

        chords = _bar_chords(
            ((off, midi, dur) for phrase in phrases for _, off, midi, dur in phrase),
            self.beats_per_bar,
        )

        for phrase in phrases:
            length = len(phrase)
            history: list[int] = []
            prev_pitch = None
            for idx, (item_index, off, midi, _duration) in enumerate(phrase):
                chord = chords.get(int(off // self.beats_per_bar))
                harmonic = _harmonic_function_bucket(
                    prev_pitch % 12 if prev_pitch is not None else None, chord
                )
                contexts[item_index] = Context(
                    history=tuple(history[-self.order :]),
                    accent=_accent_bucket(off, self.beats_per_bar),
                    phrase=_phrase_bucket(idx, length),
                    harmonic=harmonic,
                )
                if prev_pitch is not None:
                    history.append(midi - prev_pitch)
                prev_pitch = midi
        return contexts

    def line_logprob(self, line, *, use_context: bool = True) -> float:
        """Total negative log-probability (summed per-event surprise) of a line.

        ``use_context=False`` reproduces the first-order surprise exactly (the soft/optional
        contract): it is the sum of ``transition_surprise(..., context=None)`` over the line.
        """
        contexts = self._line_contexts(line) if use_context else [None] * len(line)
        total = 0.0
        previous_pitch = None
        previous_interval = 0
        previous_duration = None
        for item, context in zip(line, contexts):
            total += self.transition_surprise(
                previous_pitch, previous_interval, previous_duration, item, context=context
            )
            midi = _item_midi(item)
            if midi is not None:
                if previous_pitch is not None:
                    previous_interval = midi - previous_pitch
                previous_pitch = midi
            previous_duration = _round_duration(float(item[1]))
        return total

    def perplexity(self, lines, *, use_context: bool = True) -> float:
        """exp(mean per-event negative log-probability) over the given lines."""
        total = 0.0
        events = 0
        for line in lines:
            total += self.line_logprob(line, use_context=use_context)
            events += len(line)
        if events == 0:
            return float("inf")
        return exp(total / events)

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot of all count tables + config (keys stringified)."""
        return {
            "version": 4,
            "order": self.order,
            "beats_per_bar": self.beats_per_bar,
            "min_count": self.min_count,
            "interval_counts": {
                str(prev): {str(nxt): n for nxt, n in counts.items()}
                for prev, counts in self.interval_counts.items()
            },
            "duration_counts": {
                str(prev): {str(nxt): n for nxt, n in counts.items()}
                for prev, counts in self.duration_counts.items()
            },
            "global_intervals": {str(k): n for k, n in self.global_intervals.items()},
            "global_durations": {str(k): n for k, n in self.global_durations.items()},
            "context_interval_counts": {
                key: {str(nxt): n for nxt, n in counts.items()}
                for key, counts in self.context_interval_counts.items()
            },
            "context_duration_counts": {
                key: {str(nxt): n for nxt, n in counts.items()}
                for key, counts in self.context_duration_counts.items()
            },
            "rest_counts": dict(self.rest_counts),
            "event_counts": dict(self.event_counts),
            "global_rest": self.global_rest,
            "global_events": self.global_events,
            "rest_duration_counts": {str(k): n for k, n in self.rest_duration_counts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MarkovMelodyModel":
        interval_counts = {
            int(prev): Counter({int(nxt): n for nxt, n in counts.items()})
            for prev, counts in data["interval_counts"].items()
        }
        duration_counts = {
            float(prev): Counter({float(nxt): n for nxt, n in counts.items()})
            for prev, counts in data["duration_counts"].items()
        }
        global_intervals = Counter({int(k): n for k, n in data["global_intervals"].items()})
        global_durations = Counter({float(k): n for k, n in data["global_durations"].items()})
        # Context tables + config are v2; absent in v1 artifacts (first-order only).
        context_interval_counts = {
            key: Counter({int(nxt): n for nxt, n in counts.items()})
            for key, counts in data.get("context_interval_counts", {}).items()
        }
        context_duration_counts = {
            key: Counter({float(nxt): n for nxt, n in counts.items()})
            for key, counts in data.get("context_duration_counts", {}).items()
        }
        # Rest-as-event stats are v4; absent in v3 artifacts (rest_probability then returns 0).
        rest_duration_counts = Counter(
            {float(k): n for k, n in data.get("rest_duration_counts", {}).items()}
        )
        return cls(
            interval_counts,
            duration_counts,
            global_intervals,
            global_durations,
            context_interval_counts,
            context_duration_counts,
            dict(data.get("rest_counts", {})),
            dict(data.get("event_counts", {})),
            data.get("global_rest", 0),
            data.get("global_events", 0),
            rest_duration_counts,
            order=data.get("order", DEFAULT_ORDER),
            beats_per_bar=data.get("beats_per_bar", DEFAULT_BEATS_PER_BAR),
            min_count=data.get("min_count", DEFAULT_MIN_COUNT),
        )

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "MarkovMelodyModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
