"""Build training examples from the curated corpus.

Each example pairs a melody (encoded as key-relative Events — the CONTENT the model generates) with
its control signals (the things it is TOLD): per-bar harmony, segment shape, character tags, key.
Rejected files (the cleaning pass) are skipped; character comes from the categorization manifest.

v1 keeps it simple: one segment per piece at its global key (modulating-piece segmentation is a
later refinement, per architecture doc 18 / FE4). Per-note figuration + motif labels travel
alongside the melody as the training étude: the infilling model predicts them as auxiliary targets
(it is never told them at inference), so the figurative/recurrence structure is pressed into the
weights instead of left to chance.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from framework.analysis.events import events_from_midi, melody_line
from framework.analysis.core import Knobs
from framework.analysis.harmony import _CHORD_TONE, accompaniment, chord_at, chord_vocab, windowed_harmony
from framework.analysis.pipeline import analyze_events
from framework.analysis.segment import _archetype
from framework.analysis.tonal import estimate_key, pc_histogram
from generation.theme_nn.representation import Event, encode

ROOT = Path(__file__).resolve().parents[2]   # repo root (generation/theme_nn/dataset.py)
MANIFEST = ROOT / "trackers" / "corpus_categories.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "datasets" / "theme_nn.jsonl"


@dataclass
class TrainingExample:
    source: str
    key: tuple[int, str]
    character: list[str]      # categorization tags (primary + supporting)
    shape: str                # contour archetype
    contour: list[float]      # per-bar normalised pitch
    harmony: list[str]        # per-bar Roman numeral (the harmonic frame)
    events: list[list]        # melody: [degree(0=rest), alteration, octave, duration] per note
    figuration: list[str] = field(default_factory=list)  # per-event figure label (rest->"rest")
    motif: list[str] = field(default_factory=list)        # per-event motif transform (rest->"-")
    beat: list[int] = field(default_factory=list)         # per-event metric slot in the bar (0=downbeat)
    bar: list[int] = field(default_factory=list)          # per-event bar index from the melody start
    interval: list[int] = field(default_factory=list)     # per-note signed semitone step from prev note (99=none)
    chordpos: list[str] = field(default_factory=list)     # per-note position in the governing chord (R/3/5/7/nct)
    velocity: list[int] = field(default_factory=list)     # per-event MIDI velocity (1-127; 0 for rests)


def _load_manifest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                out[e["path"]] = e
    return out


_GRID = 1 / 12  # quantize onsets/durations to a 1/12-beat grid (keeps 16ths and triplets)
MAX_BAR = 63    # bar index is clamped here (themes past 64 bars are rare; keeps the field bounded)
BEAT_CAP = 95   # beat-slot ceiling = BEAT_SIZE-1 (covers bars up to 8 beats: 8*12); odd huge bars clamp


def _metric(onsets: list[float], bpb: float) -> tuple[list[int], list[int]]:
    """Per-event (beat-slot-in-bar, bar-index) from the REAL onset relative to the file's bar grid.

    beat slot = the onset's position on the 1/12 grid within the bar (0 = downbeat). Because the
    onsets are the file's actual onsets, an anacrusis is placed correctly: a melody picking up on
    beat 3 reads as beat-slot 36, and its first downbeat falls in bar 1, not bar 0. Bars are counted
    relative to the melody's first bar so themes are comparable wherever they sit in the file. metre
    (bpb) comes from the file's time signature; only the onset-0 default is assumed when none."""
    slots = max(1, int(round(bpb / _GRID)))
    first_bar = int(onsets[0] // bpb) if onsets else 0
    beat_out, bar_out = [], []
    for o in onsets:
        beat_out.append(min(int(round((o % bpb) / _GRID)) % slots, BEAT_CAP))
        bar_out.append(min(max(int(o // bpb) - first_bar, 0), MAX_BAR))
    return beat_out, bar_out


def _melody_with_rests(notes) -> tuple[list[tuple[int | None, float]], list[float]]:
    """Melody as (midi|None, duration) items + each item's REAL onset (beats), grid-quantized, with
    rests filled for the gaps the melody line leaves. Real onsets carry the metre phase (anacrusis)."""
    q = lambda x: round(x / _GRID) * _GRID
    items: list[tuple[int | None, float]] = []
    onsets: list[float] = []
    cursor = q(notes[0].onset)
    for n in notes:
        onset, dur = q(n.onset), max(q(n.duration), _GRID)
        if onset > cursor + _GRID / 2:
            items.append((None, round(onset - cursor, 4))); onsets.append(round(cursor, 4))
        items.append((n.midi, round(dur, 4))); onsets.append(round(onset, 4))
        cursor = onset + dur
    return items, onsets


def _contour(mel, bpb: float) -> tuple[float, ...]:
    pitches = [e.midi for e in mel]
    lo, span = min(pitches), max(max(pitches) - min(pitches), 1)
    bars: dict[int, list[int]] = {}
    for e in mel:
        bars.setdefault(int(e.onset // bpb), []).append(e.midi)
    return tuple(round((sum(bars[b]) / len(bars[b]) - lo) / span, 3) for b in sorted(bars))


def build_example(path: Path, character: list[str], knobs: Knobs,
                  min_notes: int, max_events: int) -> TrainingExample | None:
    """Melody + control frame + per-note features. Each event carries: figuration/motif (étude
    targets), beat/bar (metre, from the file's time signature, anacrusis-aware), the signed melodic
    interval from the previous note, and its position in the governing chord (R/3/5/7/nct)."""
    events, total, barlen = events_from_midi(str(path))
    if not events:
        return None
    mel = melody_line(events)
    if len(mel) < min_notes:
        return None
    key = estimate_key(pc_histogram(events))
    knobs = replace(knobs, beats_per_bar=barlen)     # use the file's real metre everywhere downstream
    bpb = barlen
    windows = windowed_harmony(accompaniment(events, mel), total, key, knobs)
    harmony = [chord_at(windows, b * bpb)[0] for b in range(int(total // bpb) + 1)]
    contour = _contour(mel, bpb)

    # Per-note étude labels, aligned to the melody notes (one per mel note, no rests yet).
    annotated = analyze_events(events, total, knobs)
    fig_seq = [n.features["figuration"]["figure"] for n in annotated]
    mot_seq = [n.features["motif"]["transform"] for n in annotated]

    items, onsets = _melody_with_rests(mel)
    encoded = encode(items, key)[:max_events]
    onsets = onsets[:len(encoded)]
    beat_out, bar_out = _metric(onsets, bpb)

    rn_root = {label: root for label, _pcs, root in chord_vocab(*key)}   # Roman numeral -> chord root pc
    ev_out, fig_out, mot_out, ivl_out, cpos_out, vel_out = [], [], [], [], [], []
    ni, prev_midi = 0, None
    for e, b in zip(encoded, bar_out):
        if e.kind == "rest":
            ev_out.append([0, e.alteration, e.octave, e.duration])
            fig_out.append("rest"); mot_out.append("-"); ivl_out.append(99); cpos_out.append("rest")
            vel_out.append(0)
        else:
            ev_out.append([e.degree, e.alteration, e.octave, e.duration])
            fig_out.append(fig_seq[ni] if ni < len(fig_seq) else "step")
            mot_out.append(mot_seq[ni] if ni < len(mot_seq) else "-")
            midi = mel[ni].midi if ni < len(mel) else (prev_midi or 60)
            ivl_out.append(99 if prev_midi is None else max(-12, min(12, midi - prev_midi)))
            root = rn_root.get(harmony[min(b, len(harmony) - 1)]) if harmony else None
            cpos_out.append(_CHORD_TONE.get((midi % 12 - root) % 12, "nct") if root is not None else "nct")
            vel_out.append(int(mel[ni].velocity) if ni < len(mel) else 64)
            prev_midi = midi
            ni += 1
    return TrainingExample(
        source=str(path.relative_to(ROOT)), key=key, character=character,
        shape=_archetype(contour), contour=list(contour), harmony=harmony,
        events=ev_out, figuration=fig_out, motif=mot_out, beat=beat_out, bar=bar_out,
        interval=ivl_out, chordpos=cpos_out, velocity=vel_out,
    )


def build_examples(limit: int | None = None, *, min_notes: int = 8, max_events: int = 400,
                   knobs: Knobs | None = None):
    """Yield TrainingExamples for every kept (non-rejected) curated file with a usable melody."""
    knobs = knobs or Knobs()
    manifest = _load_manifest()
    built = 0
    for relpath, entry in manifest.items():
        if entry.get("verdict") == "reject" or entry.get("moved_to"):
            continue
        path = ROOT / relpath
        if not path.exists():
            continue
        try:
            example = build_example(path, entry.get("primary", []) + entry.get("supporting", []),
                                    knobs, min_notes, max_events)
        except Exception:  # noqa - skip unparseable / odd files, like the corpus ingestion does
            continue
        if example is not None:
            yield example
            built += 1
            if limit and built >= limit:
                return


def worklist(min_notes: int = 8, max_events: int = 400) -> list[tuple]:
    """The (relpath, character, min_notes, max_events) build tasks for every kept, on-disk file —
    the unit of work for a parallel build."""
    return [(relpath, entry.get("primary", []) + entry.get("supporting", []), min_notes, max_events)
            for relpath, entry in _load_manifest().items()
            if entry.get("verdict") != "reject" and not entry.get("moved_to")
            and (ROOT / relpath).exists()]


def build_entry(task: tuple) -> "TrainingExample | None":
    """Picklable worker: build one example from a ``worklist`` task (per-file, so it parallelises
    cleanly across processes). Swallows per-file parse errors like the serial path does."""
    relpath, character, min_notes, max_events = task
    try:
        return build_example(ROOT / relpath, character, Knobs(), min_notes, max_events)
    except Exception:  # noqa - skip unparseable / odd files
        return None


def save_jsonl(examples, out: Path = DEFAULT_OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for example in examples:
            if example is not None:
                f.write(json.dumps(asdict(example)) + "\n")
    return out
