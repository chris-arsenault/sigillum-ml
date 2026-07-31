"""Meter-normalized multi-family onset roll -- the "image" the whole-score denoiser works on.

A whole score is projected by Ruby Partitura into ``timed_events`` (see
``outputs/datasets/whole_score``). Here we turn a window of that score into a dense
``(CHANNELS, STEPS)`` float grid so a 1-D diffusion model can corrupt and denoise it.

Layout
------
* **Time is meter-normalized.** Every measure is divided into ``SUBDIV`` equal steps
  regardless of its time signature, so a downbeat is always a step boundary and phrase
  structure lines up across pieces in different meters. A window is ``WINDOW_MEASURES``
  measures -> ``STEPS = WINDOW_MEASURES * SUBDIV`` columns.
* **Pitch x instrument family are the channels.** We use ``PITCH_COUNT`` chromatic pitches
  starting at ``LOW_MIDI`` for each of ``FAMILIES`` instrument families, giving
  ``CHANNELS = len(FAMILIES) * PITCH_COUNT`` rows. Cell ``(family*PITCH_COUNT + pitch, step)``
  is ``1.0`` when an instrument of that family has a note **onset** of that pitch at that step.

It is an *onset* roll (only the first cell of each note fires), which keeps the grid sparse
(~0.4% active) and makes the representation invertible enough to hand structured events back to
Ruby Partitura for materialization. This module owns no musical semantics beyond bucketing --
it never parses MusicXML and never renders audio; Partitura remains the score authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

import numpy as np

# --- grid geometry -------------------------------------------------------------------
LOW_MIDI = 21            # A0
PITCH_COUNT = 88         # A0 .. C8 (covers the observed corpus range 22..106)
HIGH_MIDI = LOW_MIDI + PITCH_COUNT - 1
SUBDIV = 16              # steps per measure (meter-normalized: 16th-note resolution in 4/4)
WINDOW_MEASURES = 8
STEPS = WINDOW_MEASURES * SUBDIV

# --- instrument families -------------------------------------------------------------
# Twelve coarse orchestral families. Keyword classification of the Partitura part name/
# abbreviation; the first family whose keyword matches wins, so order matters (specific
# before generic). Unclassifiable parts fall back to FALLBACK_FAMILY and are counted so the
# dataset can report coverage honestly.
FAMILIES: tuple[str, ...] = (
    "flute",
    "oboe",
    "clarinet",
    "bassoon",
    "horn",
    "trumpet",
    "trombone",
    "tuba",
    "percussion",
    "violin",
    "viola",
    "low_strings",
)
FAMILY_INDEX = {name: i for i, name in enumerate(FAMILIES)}
FALLBACK_FAMILY = "low_strings"
CHANNELS = len(FAMILIES) * PITCH_COUNT

# Keyword -> family. Checked in this order; substrings are matched case-insensitively against
# the lowercased, alphanumeric-only part name and abbreviation.
_FAMILY_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("piccolo", "flute"),
    ("flaut", "flute"),
    ("flote", "flute"),
    ("flauto", "flute"),
    ("flute", "flute"),
    ("englishhorn", "oboe"),
    ("coranglais", "oboe"),
    ("oboe", "oboe"),
    ("hautbois", "oboe"),
    ("bassclarinet", "clarinet"),
    ("clarinet", "clarinet"),
    ("klarinet", "clarinet"),
    ("contrabassoon", "bassoon"),
    ("contrafagott", "bassoon"),
    ("bassoon", "bassoon"),
    ("fagott", "bassoon"),
    ("basson", "bassoon"),
    ("horn", "horn"),
    ("corno", "horn"),
    ("cornet", "trumpet"),
    ("trumpet", "trumpet"),
    ("tromba", "trumpet"),
    ("trompet", "trumpet"),
    ("basstrombone", "trombone"),
    ("trombone", "trombone"),
    ("posaune", "trombone"),
    ("tuba", "tuba"),
    ("timpani", "percussion"),
    ("timbales", "percussion"),
    ("percussion", "percussion"),
    ("cymbal", "percussion"),
    ("triangle", "percussion"),
    ("snare", "percussion"),
    ("drum", "percussion"),
    ("violoncello", "low_strings"),
    ("cello", "low_strings"),
    ("contrabass", "low_strings"),
    ("doublebass", "low_strings"),
    ("continuo", "low_strings"),
    ("violino", "violin"),
    ("violin", "violin"),
    ("viola", "viola"),
    ("bratsche", "viola"),
)


def classify_family(*names: str) -> str:
    """Return the instrument family for a part, from its name/abbreviation keywords.

    Falls back to ``FALLBACK_FAMILY`` when nothing matches.
    """
    blob = "".join(n for n in names if n).lower()
    blob = "".join(ch for ch in blob if ch.isalnum())
    for keyword, family in _FAMILY_KEYWORDS:
        if keyword in blob:
            return family
    return FALLBACK_FAMILY


def part_families(score: dict) -> dict:
    """Map ``part_id -> family`` for every part in a Partitura score observation."""
    out: dict = {}
    for part in score.get("parts", []):
        names = [part.get("name", ""), part.get("abbreviation", "")]
        for inst in part.get("instruments", []) or []:
            names.append(inst.get("name", ""))
        out[part["id"]] = classify_family(*names)
    return out


@dataclass(frozen=True)
class MeasureSpan:
    index: int
    offset_ql: Fraction
    duration_ql: Fraction


def _frac(value) -> Fraction:
    return Fraction(str(value))


def measure_spans(score: dict) -> list:
    """Ordered, de-duplicated measure spans (by ``index``) with rational offsets/durations."""
    seen: dict = {}
    for m in score.get("measures", []):
        idx = int(m["index"])
        if idx not in seen:
            seen[idx] = MeasureSpan(idx, _frac(m["offset_ql"]), _frac(m["duration_ql"]))
    return [seen[k] for k in sorted(seen)]


def _measure_step(measure_onset_ql: Fraction, measure_duration_ql: Fraction) -> int:
    """Meter-normalized step within a measure: position ratio -> [0, SUBDIV)."""
    if measure_duration_ql <= 0:
        return 0
    ratio = measure_onset_ql / measure_duration_ql
    step = int(ratio * SUBDIV)
    return max(0, min(SUBDIV - 1, step))


def channel_for(family: str, midi: int):
    """Channel row for ``(family, midi)``; ``None`` if the pitch is out of the grid range."""
    if midi < LOW_MIDI or midi > HIGH_MIDI:
        return None
    fam = FAMILY_INDEX.get(family, FAMILY_INDEX[FALLBACK_FAMILY])
    return fam * PITCH_COUNT + (midi - LOW_MIDI)


def build_window_roll(score: dict, start_measure: int, *, families: dict = None, n_measures: int = WINDOW_MEASURES) -> np.ndarray:
    """Render ``n_measures`` starting at measure ``start_measure`` (inclusive) into a
    ``(CHANNELS, n_measures*SUBDIV)`` onset roll. Out-of-range pitches are dropped."""
    families = families if families is not None else part_families(score)
    steps = n_measures * SUBDIV
    roll = np.zeros((CHANNELS, steps), dtype=np.float32)
    span_by_index = {m.index: m for m in measure_spans(score)}
    window = set(range(start_measure, start_measure + n_measures))
    for ev in score.get("timed_events", []):
        if ev.get("kind") != "note" or ev.get("midi") is None:
            continue
        mi = int(ev["measure_index"])
        if mi not in window:
            continue
        span = span_by_index.get(mi)
        if span is None:
            continue
        ch = channel_for(families.get(ev["part_id"], FALLBACK_FAMILY), int(ev["midi"]))
        if ch is None:
            continue
        local = (mi - start_measure) * SUBDIV + _measure_step(_frac(ev["measure_onset_ql"]), span.duration_ql)
        roll[ch, local] = 1.0
    return roll


def roll_to_events(roll: np.ndarray, *, start_measure: int = 1, threshold: float = 0.5) -> list:
    """Inverse of :func:`build_window_roll`: active cells -> structured onset events.

    Returns dicts ``{family, midi, measure_index, step_in_measure, subdivisions}`` -- enough for a
    Ruby Partitura bridge to place notes on a meter grid. This is intentionally lossy: exact part,
    duration, and voice are not recovered (an onset roll does not carry them).
    """
    events: list = []
    rows, cols = np.where(roll > threshold)
    for ch, step in zip(rows.tolist(), cols.tolist()):
        family = FAMILIES[ch // PITCH_COUNT]
        midi = LOW_MIDI + (ch % PITCH_COUNT)
        measure_index = start_measure + step // SUBDIV
        events.append(
            {
                "family": family,
                "midi": midi,
                "measure_index": measure_index,
                "step_in_measure": step % SUBDIV,
                "subdivisions": SUBDIV,
            }
        )
    events.sort(key=lambda e: (e["measure_index"], e["step_in_measure"], e["midi"]))
    return events


def family_time_activity(roll: np.ndarray) -> np.ndarray:
    """Collapse pitch within each family: ``(len(FAMILIES), STEPS)`` onset counts per family/step."""
    steps = roll.shape[1]
    per_family = roll.reshape(len(FAMILIES), PITCH_COUNT, steps)
    return per_family.sum(axis=1)


def density(roll: np.ndarray) -> float:
    """Fraction of active cells -- the sparsity/activity of a roll."""
    return float(roll.mean())


def score_windows(score: dict, *, n_measures: int = WINDOW_MEASURES, stride: int = None):
    """Yield valid ``start_measure`` values for non-truncated windows over a score."""
    spans = measure_spans(score)
    if not spans:
        return
    stride = stride if stride is not None else n_measures
    first = spans[0].index
    last = spans[-1].index
    start = first
    while start + n_measures - 1 <= last:
        yield start
        start += stride
