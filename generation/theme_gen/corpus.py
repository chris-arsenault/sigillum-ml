"""Training corpus sources for the Markov melody model.

Currently the repo-local fallback corpus. Decision D1
(``docs/architecture/17_theme_generator.md``) grows this into a pluggable interface with
MIDI/MusicXML ingestion and a music21 bundled default (plan M1). The ``themes`` import is
kept lazy so importing this module never triggers theme-load side effects.
"""
from __future__ import annotations

import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from framework.foundation import paths as _paths
from generation.theme_gen._common import MelodyItem

_INGEST_SUFFIXES = (".mid", ".midi", ".musicxml", ".xml")

# Default "general tonal" set from music21's bundled public-domain corpus (decision D1):
# Bach chorales (functional tonality) + Essen folksong + fiddle/dance tunes (Ryan's Mammoth
# and O'Neill's 1850) for singable, strongly melodic tonal/modal lines. Modern material
# (game/trance) and maqam/Arabian material are licensing-bound and not bundled; they enter
# through the ``ingest`` source and, for distinct styles, separate models.
_DEFAULT_MUSIC21_CORPORA = ("bach", "essenFolksong", "ryansMammoth", "oneills1850")

# Per-source cap when sampling the default corpus, so each source contributes (balance) and
# the build stays bounded. Swappable.
DEFAULT_PER_CORPUS_LIMIT = 120


def ingest_file(path) -> list[MelodyItem] | None:
    """Parse one MIDI/MusicXML file into a monophonic melody item-list, or None.

    MIDI is parsed at the channel level (``mido``) because music21 collapses channels into
    one part, which hides the melody on single-track / single-part rips. Each (track,
    channel) voice gets a skyline (top sounding note over time); the most melody-like voice
    wins (see ``_score_melody_line``). MusicXML routes through music21. Returns None on
    failure or when no usable melody is found.
    """
    path = Path(path)
    if path.suffix.lower() in (".mid", ".midi"):
        return _ingest_midi(path)

    from music21 import converter

    try:
        score = converter.parse(str(path))
    except Exception as exc:  # music21 raises a variety of types on bad input
        warnings.warn(f"theme_gen.corpus: could not parse {path}: {exc}")
        return None
    return _extract_line(score, str(path))


_DRUM_CHANNEL = 9  # General MIDI percussion (channel 10, zero-indexed)
_QUANTIZE_GRID = 1 / 12.0  # round durations to this grid (keeps 16ths and 8th-triplets)


def _collect_midi_voices(mid) -> dict:
    """Group note (start_tick, end_tick, pitch) by (track, channel); drums excluded."""
    voices: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for track_index, track in enumerate(mid.tracks):
        clock = 0
        ongoing: dict[tuple[int, int], list[int]] = defaultdict(list)
        for msg in track:
            clock += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                ongoing[(msg.channel, msg.note)].append(clock)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                starts = ongoing.get((msg.channel, msg.note))
                if starts:
                    start = starts.pop(0)
                    if msg.channel != _DRUM_CHANNEL and clock > start:
                        voices[(track_index, msg.channel)].append((start, clock, msg.note))
    return voices


def _voice_skyline(notes, ticks_per_beat) -> tuple[list[tuple[int | None, float]], float]:
    """Monophonic skyline (top sounding pitch over time, with rests) + polyphony fraction."""
    points = sorted({tick for start, end, _ in notes for tick in (start, end)})
    segments: list[tuple[int | None, int, int]] = []
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        active = [pitch for start, end, pitch in notes if start <= left < end]
        segments.append((max(active) if active else None, right - left, len(active)))

    total = sum(dur for _, dur, _ in segments)
    chordal = sum(dur for _, dur, n in segments if n > 1)
    polyphony = chordal / total if total else 0.0

    merged: list[list] = []
    for pitch, dur, _ in segments:
        ql = dur / ticks_per_beat
        if merged and merged[-1][0] == pitch:
            merged[-1][1] += ql
        else:
            merged.append([pitch, ql])
    return [(pitch, ql) for pitch, ql in merged], polyphony


def _score_melody_line(line, polyphony: float) -> float | None:
    """Melody-likeness of a monophonic skyline line (register + variety - polyphony)."""
    midis = [pitch for pitch, _ in line if pitch is not None]
    if len(midis) < MIN_MELODY_NOTES:
        return None
    pitch_classes = len({midi % 12 for midi in midis})
    if pitch_classes < MIN_MELODY_PITCH_CLASSES or _too_leapy(midis):
        return None
    mean_pitch = sum(midis) / len(midis)
    return (
        _MELODY_REGISTER_WEIGHT * mean_pitch
        + _MELODY_VARIETY_WEIGHT * pitch_classes
        - _MELODY_POLYPHONY_PENALTY * polyphony
    )


def _ingest_midi(path: Path) -> list[MelodyItem] | None:
    import mido
    from music21 import pitch as m21pitch

    try:
        mid = mido.MidiFile(str(path))
    except Exception as exc:
        warnings.warn(f"theme_gen.corpus: could not parse {path}: {exc}")
        return None
    ticks_per_beat = mid.ticks_per_beat or 480

    best_line = None
    best_score = None
    for notes in _collect_midi_voices(mid).values():
        if not notes:
            continue
        line, polyphony = _voice_skyline(notes, ticks_per_beat)
        score = _score_melody_line(line, polyphony)
        if score is not None and (best_score is None or score > best_score):
            best_line, best_score = line, score

    if best_line is None:
        warnings.warn(f"theme_gen.corpus: no usable melody in {path}")
        return None

    items: list[MelodyItem] = []
    for pitch, ql in best_line:
        quantized = round(ql / _QUANTIZE_GRID) * _QUANTIZE_GRID
        if quantized < _QUANTIZE_GRID / 2:
            continue
        if pitch is None:
            items.append((None, quantized))
        else:
            spelled = m21pitch.Pitch()
            spelled.midi = pitch
            items.append((spelled.nameWithOctave, quantized))
    if not any(item[0] is not None for item in items):
        warnings.warn(f"theme_gen.corpus: no usable melody in {path}")
        return None
    return items


# Melody-track selection heuristics (swappable). Multi-track MIDI (e.g. NES VGM) separates
# lead / bass / arpeggio / drums; the melody is rarely the densest track. We score each
# pitched part and pick the most melody-like: higher (non-bass) register, more pitch-class
# variety, and lower polyphony (a lead is near-monophonic, not a chord pad or drum track).
MIN_MELODY_NOTES = 6
MIN_MELODY_PITCH_CLASSES = 3
# Reject a line whose intervals leap more than an octave too often: that is the signature
# of melody + bass interleaved on one voice (not a singable line). Swappable.
MAX_BIG_LEAP_FRACTION = 0.15
_BIG_LEAP = 12
_MELODY_REGISTER_WEIGHT = 1.0
_MELODY_VARIETY_WEIGHT = 3.0
# Polyphony is only a gentle tiebreaker: a harmonized lead is still the melody (the skyline
# takes its top line), so we must not penalize it out of contention.
_MELODY_POLYPHONY_PENALTY = 8.0


def _too_leapy(midis: Sequence[int]) -> bool:
    intervals = [abs(midis[i + 1] - midis[i]) for i in range(len(midis) - 1)]
    if not intervals:
        return False
    return sum(1 for value in intervals if value > _BIG_LEAP) / len(intervals) > MAX_BIG_LEAP_FRACTION


def _part_melody_line(part) -> tuple[list[MelodyItem], float] | None:
    """A part's monophonic top-line item-list + a melody-likeness score, or None if the
    part is too sparse / static / unpitched (a drum or pad track)."""
    items: list[MelodyItem] = []
    midis: list[int] = []
    chords = 0
    for element in part.flatten().notesAndRests:
        ql = float(element.quarterLength)
        if element.isRest:
            items.append((None, ql))
        elif element.isChord:
            chords += 1
            top = max(element.pitches, key=lambda pitch: pitch.ps)
            items.append((top.nameWithOctave, ql))
            midis.append(int(top.midi))
        elif getattr(element, "isNote", False) and getattr(element, "pitch", None) is not None:
            items.append((element.nameWithOctave, ql))
            midis.append(int(element.pitch.midi))
        # unpitched percussion notes are dropped (they leave a gap, not a pitch)

    if len(midis) < MIN_MELODY_NOTES:
        return None
    pitch_classes = len({midi % 12 for midi in midis})
    if pitch_classes < MIN_MELODY_PITCH_CLASSES or _too_leapy(midis):
        return None  # static drone, percussion, or melody+bass interleaved
    mean_pitch = sum(midis) / len(midis)
    polyphony = chords / len(midis)
    score = (
        _MELODY_REGISTER_WEIGHT * mean_pitch
        + _MELODY_VARIETY_WEIGHT * pitch_classes
        - _MELODY_POLYPHONY_PENALTY * polyphony
    )
    return items, score


def _extract_line(score, label: str) -> list[MelodyItem] | None:
    """Reduce a parsed music21 score to one monophonic melody item-list, or None.

    Scores every pitched part and keeps the most melody-like (register / variety /
    monophonicity), then returns that part's top-line as ``(name, ql)`` / ``(None, ql)``.
    """
    parts = list(getattr(score, "parts", []) or [])
    sources = parts if parts else [score]

    best_items: list[MelodyItem] | None = None
    best_score: float | None = None
    for part in sources:
        result = _part_melody_line(part)
        if result is None:
            continue
        items, score_value = result
        if best_score is None or score_value > best_score:
            best_items, best_score = items, score_value

    if best_items is None or not any(item[0] is not None for item in best_items):
        warnings.warn(f"theme_gen.corpus: no usable melody in {label}")
        return None
    return best_items


def ingest_dir(directory, *, recursive: bool = False, limit=None, seed: int = 0) -> tuple[list[MelodyItem], ...]:
    """Ingest MIDI/MusicXML files in ``directory``, skipping unusable ones.

    ``recursive=True`` walks subfolders too (the nested category dirs ``exotic/<scale>/``,
    ``vgm/<game>/``). ``limit`` samples that many files (seeded) BEFORE parsing, so a cap is
    cheap. A missing directory yields an empty corpus.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return ()
    walker = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(f for f in walker if f.is_file() and f.suffix.lower() in _INGEST_SUFFIXES)
    if limit is not None and len(files) > limit:
        files = sorted(random.Random(seed).sample(files, limit))
    lines: list[list[MelodyItem]] = []
    for file in files:
        line = ingest_file(file)
        if line is not None:
            lines.append(line)
    return tuple(lines)


def default_theme_corpus() -> tuple[Sequence[MelodyItem], ...]:
    """Use the repo's current melodic material as the local fallback corpus."""

    try:                                   # the piece's themes are an opportunistic seed, not a dependency
        from symphony.materials import themes
    except ModuleNotFoundError:
        return ()
    return (themes.RENN_FULL, themes.ESHAIA_FULL, themes.ONSLAUGHT_PULSE, themes.ONSLAUGHT_B,
            themes.MEMORY_FULL, themes.MEMORY_MEM)


# music21 composers selectable by name; "fiddle" is an alias for the dance-tune sets.
_MUSIC21_NAMES = ("bach", "essenFolksong", "ryansMammoth", "oneills1850",
                  "palestrina", "monteverdi", "trecento")
_FIDDLE = ("ryansMammoth", "oneills1850")
_BIG = 10 ** 9  # "take all" for per-corpus music21 selection


def load_corpus(*sources: str, sample_seed: int = 0) -> tuple[Sequence[MelodyItem], ...]:
    """Load and concatenate melodic lines from a composable corpus spec.

    Each token is ``name`` or ``name@N`` (cap to N melodies, sampled with ``sample_seed``):

      themes                  repo-local locked themes
      music21                 the default general-tonal bundle (bach + essen + fiddle)
      bach / essenFolksong / ryansMammoth / oneills1850 / palestrina / ...   one music21 set
      fiddle                  ryansMammoth + oneills1850
      folk                    essenFolksong, fully expanded (every song in each opus)
      vgm / exotic / classical / arabian / <subdir>   ingested category under assets/raw/corpus/
      ingest                  ingested files in assets/raw/corpus/ (top level only)
      ingest:<subdir>         ingested files under assets/raw/corpus/<subdir>/ (recursive)

    Examples: ``load_corpus("exotic", "fiddle@150")`` or ``load_corpus("vgm", "bach", "classical")``.
    With no sources, defaults to ``themes``.
    """
    if not sources:
        sources = ("themes",)
    lines: list[Sequence[MelodyItem]] = []
    for token in sources:
        lines.extend(_resolve_source(token, sample_seed))
    return tuple(lines)


def _resolve_source(token: str, sample_seed: int) -> tuple[Sequence[MelodyItem], ...]:
    name, _, cap_str = token.partition("@")
    cap = int(cap_str) if cap_str else None

    if name == "themes":
        lines = list(default_theme_corpus())
        if cap is not None and len(lines) > cap:
            lines = random.Random(sample_seed).sample(lines, cap)
        return tuple(lines)
    if name == "music21":
        return music21_corpus(limit=cap or _BIG, per_corpus=cap is None)
    if name == "fiddle":
        return music21_corpus(_FIDDLE, cap or _BIG, per_corpus=cap is None)
    if name in ("folk", "essen"):
        return _essen_songs(cap)
    if name in _MUSIC21_NAMES:
        return music21_corpus((name,), cap or _BIG, per_corpus=cap is None)
    if name == "ingest":
        return ingest_dir(_paths.RAW_CORPUS, recursive=False, limit=cap, seed=sample_seed)
    if name.startswith("ingest:"):
        sub = name.split(":", 1)[1]
        return ingest_dir(_paths.RAW_CORPUS / sub, recursive=True, limit=cap, seed=sample_seed)
    directory = _paths.RAW_CORPUS / name  # a category subfolder (vgm/exotic/classical/...)
    if directory.is_dir():
        return ingest_dir(directory, recursive=True, limit=cap, seed=sample_seed)
    raise ValueError(f"unknown corpus source: {name!r}")


def _essen_songs(limit=None) -> tuple[Sequence[MelodyItem], ...]:
    """Folk songs from the Essen collection (each file is an opus of many songs)."""
    from music21 import corpus as m21corpus

    lines: list[Sequence[MelodyItem]] = []
    for work in m21corpus.getComposer("essenFolksong"):
        if limit is not None and len(lines) >= limit:
            break
        try:
            opus = m21corpus.parse(work)
        except Exception as exc:
            warnings.warn(f"theme_gen.corpus: could not parse {work}: {exc}")
            continue
        scores = list(getattr(opus, "scores", []) or []) or [opus]
        for score in scores:
            if limit is not None and len(lines) >= limit:
                break
            line = _extract_line(score, str(work))
            if line is not None:
                lines.append(line)
    return tuple(lines)


def music21_corpus(
    corpora: Sequence[str] = _DEFAULT_MUSIC21_CORPORA,
    limit: int = 20,
    *,
    per_corpus: bool = False,
) -> tuple[Sequence[MelodyItem], ...]:
    """Melodic lines from music21's bundled public-domain corpus (decision D1).

    Extracts one monophonic line per parsed work. ``limit`` caps the number of works: a
    total cap by default, or a per-source cap when ``per_corpus=True`` (so every source in
    ``corpora`` contributes — used when building the persisted general-tonal model).
    Unparseable works are skipped.
    """
    from music21 import corpus as m21corpus

    lines: list[Sequence[MelodyItem]] = []
    for name in corpora:
        if not per_corpus and len(lines) >= limit:
            break
        taken = 0
        for work in m21corpus.getComposer(name):
            if taken >= limit or (not per_corpus and len(lines) >= limit):
                break
            try:
                score = m21corpus.parse(work)
            except Exception as exc:
                warnings.warn(f"theme_gen.corpus: could not parse {work}: {exc}")
                continue
            line = _extract_line(score, str(work))
            if line is not None:
                lines.append(line)
                taken += 1
    return tuple(lines)
