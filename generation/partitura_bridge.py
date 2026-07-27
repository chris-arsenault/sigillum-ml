"""Small process boundary from Python generation code to the Ruby Partitura CLI.

Sigillum ML may propose sounding material, but Partitura owns the executable
score source, validation, and MusicXML/MIDI rendering. Generated sources contain
fully materialized event lists; they do not hide musical repetition in Ruby
helpers or loops.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from music21 import pitch as m21pitch


class PartituraBridgeError(RuntimeError):
    """Partitura rejected a generated score source or failed to export it."""


@dataclass(frozen=True)
class PartituraScoreSource:
    """A complete, explicit Partitura Ruby source ready for validation/export."""

    ruby: str
    title: str


def _ruby_string(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _duration_token(value: Any) -> str:
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise PartituraBridgeError(f"duration must be positive and finite, got {value!r}")
    return format(duration, ".12g")


def _pitch_token(value: Any) -> str:
    if value is None:
        return "r"
    if isinstance(value, (list, tuple)):
        if not value:
            raise PartituraBridgeError("a chord must contain at least one pitch")
        return "[" + ",".join(_pitch_token(pitch) for pitch in value) + "]"
    try:
        parsed = m21pitch.Pitch(value)
    except Exception as error:
        raise PartituraBridgeError(f"invalid pitch {value!r}") from error
    return parsed.nameWithOctave.replace("-", "b")


def _event_token(item: Sequence[Any]) -> str:
    if len(item) < 2:
        raise PartituraBridgeError(f"event needs pitch and duration, got {item!r}")
    event = f"{_pitch_token(item[0])}:{_duration_token(item[1])}"
    marks = []
    for value in item[2:]:
        mark = str(value)
        if not re.fullmatch(r"[^,{}\s]+", mark):
            raise PartituraBridgeError(f"invalid inline mark {value!r}")
        marks.append(mark)
    if marks:
        event += "{" + ",".join(marks) + "}"
    return event


def _events_with_barlines(
    items: Sequence[Sequence[Any]], beats_per_bar: float
) -> str:
    tokens: list[str] = []
    cursor = 0.0
    for index, item in enumerate(items):
        duration = float(item[1])
        remaining = duration
        continued_from_previous_bar = False
        while remaining > 1e-9:
            position = cursor % beats_per_bar
            if position < 1e-9 or beats_per_bar - position < 1e-9:
                position = 0.0
            available = beats_per_bar - position
            chunk = min(remaining, available)
            continues_into_next_bar = remaining - chunk > 1e-9
            chunk_item = [item[0], chunk]
            if not continued_from_previous_bar:
                chunk_item.extend(item[2:])
            if item[0] is not None:
                if continued_from_previous_bar:
                    chunk_item.append("tie)")
                if continues_into_next_bar:
                    chunk_item.append("tie(")
            tokens.append(_event_token(chunk_item))
            cursor += chunk
            remaining -= chunk
            if (
                abs(cursor % beats_per_bar) < 1e-9
                and (remaining > 1e-9 or index < len(items) - 1)
            ):
                tokens.append("|")
            continued_from_previous_bar = True
    return " ".join(tokens)


def single_part_score(
    *,
    title: str,
    items: Sequence[Sequence[Any]],
    meter: str,
    key: str,
    tempo: int,
    beats_per_bar: float,
) -> PartituraScoreSource:
    """Materialize a generated monophonic line as one Partitura source."""

    if not items:
        raise PartituraBridgeError("cannot build a score from an empty event list")
    if beats_per_bar <= 0:
        raise PartituraBridgeError("beats_per_bar must be positive")
    total = sum(float(item[1]) for item in items)
    bars = max(1, math.ceil(total / beats_per_bar))
    remainder = bars * beats_per_bar - total
    materialized = list(items)
    if remainder > 1e-9:
        materialized.append((None, remainder))
    events = _events_with_barlines(materialized, beats_per_bar)
    source = "\n".join(
        (
            f"production_piece {_ruby_string(title)}, id: :generated_score do",
            f"  meter {_ruby_string(meter)}",
            f"  key {_ruby_string(key)}",
            f"  tempo {_ruby_string(f'quarter = {int(tempo)}')}",
            "",
            "  roster do",
            '    part :theme, "Theme", music21: "Flute", family: :woodwind',
            "  end",
            "",
            f'  section :survey, "Generated material", bars: 1..{bars} do',
            f"    span :survey_span, bars: 1..{bars} do",
            f"      phrase(:theme_line, surface: :absolute) {{ events {_ruby_string(events)} }}",
            '      placement :theme_line, part: :theme, at: "bar 1 beat 1", role: :foreground',
            "    end",
            "  end",
            "end",
            "",
        )
    )
    return PartituraScoreSource(ruby=source, title=title)


def export_score(
    score: PartituraScoreSource,
    out_dir: str | Path,
    stem: str,
    *,
    partitura_bin: str | Path | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[Path, Path]:
    """Validate and export a generated source through Partitura."""

    project_root = Path(__file__).resolve().parents[1]
    executable = Path(
        partitura_bin
        or project_root.parent / "sigillum-library" / "partitura" / "bin" / "partitura"
    ).resolve()
    if not executable.is_file():
        raise PartituraBridgeError(f"Partitura executable does not exist: {executable}")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sigillum-generated-score-") as temp_name:
        source_path = Path(temp_name) / "generated_score.rb"
        source_path.write_text(score.ruby, encoding="utf-8")
        command = subprocess.run(
            ("ruby", str(executable), "export", str(source_path), "--stem", stem),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if command.returncode != 0:
            detail = command.stderr.strip() or command.stdout.strip() or "unknown error"
            raise PartituraBridgeError(f"Partitura export failed: {detail}")
        try:
            response = json.loads(command.stdout)
            rendered_xml = Path(response["musicxml"])
            rendered_midi = Path(response["midi"])
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise PartituraBridgeError(
                "Partitura export did not return MusicXML/MIDI paths"
            ) from error
        if not rendered_xml.is_file() or not rendered_midi.is_file():
            raise PartituraBridgeError("Partitura export artifacts are missing")
        xml = destination / f"{stem}.musicxml"
        midi = destination / f"{stem}.mid"
        shutil.copyfile(rendered_xml, xml)
        shutil.copyfile(rendered_midi, midi)
    return xml, midi
