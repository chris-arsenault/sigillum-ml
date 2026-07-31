"""Build per-bar harmonic progressions from Partitura-projected corpus data.

This is the harmony instantiation of the resolution ladder. It reduces the
Ruby-projected annotation observations into a per-bar downbeat chord grid: for
each measure, the harmonic-function annotation sounding at that measure's
downbeat, expressed as a key-relative Roman-numeral function token (for example
``I``, ``V7``, ``ii65``, ``V/V``).

The token is deliberately key-relative rather than absolute (no ``C:`` prefix):
harmonic *function* transfers across composers and keys, so a held-out composer
can share a vocabulary with the training composers. The annotators re-anchor the
local key at modulations, so a run of Roman numerals still encodes real
functional motion including tonicizations.

This is dataset manufacture over already-projected facts, not a second score
runtime. Python does not parse MusicXML or invent sounding semantics here: the
chord symbols, local keys, measure indices, and quarter-length offsets all come
from Partitura observations. The per-bar downbeat reduction is a deliberate,
stated coarsening of harmonic rhythm for the first ladder level.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


NONE_TOKEN = "<none>"
HARMONIC_FUNCTION_TARGET = "harmonic_function"


def normalize_roman(roman: str) -> str:
    """Collapse a figured Roman numeral to its functional class.

    Inversions and figured-bass digits (``I6``, ``I64``, ``V65``, ``V43``,
    ``viio7``) are surface detail for a finer ladder level; the functional level
    keeps only the scale degree, quality, applied target, and accidental (``V7``
    -> ``V``, ``ii65`` -> ``ii``, ``V7/V`` -> ``V/V``, ``I64`` -> ``I``).
    """

    def _functional(part: str) -> str:
        stripped = re.sub(r"[0-9]", "", part).strip()
        stripped = stripped.replace("^", "")
        return stripped or part.strip()

    roman = roman.strip()
    if "/" in roman:
        base, applied = roman.split("/", 1)
        return f"{_functional(base)}/{_functional(applied)}"
    return _functional(roman)


class HarmonyCorpusError(ValueError):
    """Raised when the harmonic-function corpus cannot be assembled."""


@dataclass(frozen=True)
class HarmonyProgression:
    """A per-bar downbeat chord progression for one score.

    ``tokens[i]`` is the key-relative Roman functional chord at the downbeat of
    measure ``first_measure + i`` (for example ``I``, ``V``, ``V/V``), or
    ``<none>`` when no annotation covers that downbeat. ``home_key`` is the most
    common local key across the annotated span.
    """

    score_id: str
    lineage_id: str
    split: str
    home_key: str
    first_measure: int
    tokens: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def annotated_fraction(self) -> float:
        if not self.tokens:
            return 0.0
        annotated = sum(1 for token in self.tokens if token != NONE_TOKEN)
        return annotated / len(self.tokens)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarmonyCorpusError(f"cannot read {path}: {error}") from error


def _measure_offsets(score_observation: dict[str, Any]) -> dict[int, Fraction]:
    measures = score_observation.get("score", {}).get("measures", [])
    offsets: dict[int, Fraction] = {}
    for measure in measures:
        offsets[int(measure["index"])] = Fraction(str(measure["offset_ql"]))
    if not offsets:
        raise HarmonyCorpusError("score observation has no measures")
    return offsets


def _chord_token(example: dict[str, Any]) -> str | None:
    metadata = example.get("metadata", {})
    roman = metadata.get("roman_number")
    if not roman:
        return None
    return normalize_roman(str(roman))


def _local_key(example: dict[str, Any]) -> str | None:
    return example.get("metadata", {}).get("key")


def _progression_tokens(
    examples: list[dict[str, Any]],
    offsets: dict[int, Fraction],
) -> tuple[int, list[str], list[str | None]]:
    """Assign each measure the chord sounding at its downbeat."""

    spans: list[tuple[Fraction, Fraction, str, str | None]] = []
    for example in examples:
        if example.get("target") != HARMONIC_FUNCTION_TARGET:
            continue
        token = _chord_token(example)
        if token is None:
            continue
        scope = example["scope"]
        start = Fraction(str(scope["start_ql"]))
        end = Fraction(str(scope["end_ql"]))
        if end <= start:
            continue
        spans.append((start, end, token, _local_key(example)))
    if not spans:
        raise HarmonyCorpusError("score has no usable harmonic-function spans")
    spans.sort(key=lambda item: item[0])
    first_measure = min(offsets)
    last_measure = max(offsets)
    tokens: list[str] = []
    keys: list[str | None] = []
    for measure in range(first_measure, last_measure + 1):
        downbeat = offsets[measure]
        active = NONE_TOKEN
        active_key: str | None = None
        for start, end, token, local_key in spans:
            if start <= downbeat < end:
                active = token
                active_key = local_key
            elif start > downbeat:
                break
        tokens.append(active)
        keys.append(active_key)
    return first_measure, tokens, keys


def _trim_none_edges(
    first_measure: int, tokens: list[str], keys: list[str | None]
) -> tuple[int, list[str], list[str | None]]:
    start = 0
    while start < len(tokens) and tokens[start] == NONE_TOKEN:
        start += 1
    end = len(tokens)
    while end > start and tokens[end - 1] == NONE_TOKEN:
        end -= 1
    return first_measure + start, tokens[start:end], keys[start:end]


def _home_key(keys: list[str | None]) -> str:
    from collections import Counter

    counts = Counter(key for key in keys if key)
    if not counts:
        return "C"
    return counts.most_common(1)[0][0]


def build_progressions(
    annotation_manifest: Path,
    pilot_manifest: Path,
) -> list[HarmonyProgression]:
    """Assemble per-bar progressions for every score carrying harmonic function."""

    annotation_manifest = Path(annotation_manifest)
    pilot_manifest = Path(pilot_manifest)
    annotation_root = annotation_manifest.parent
    pilot_root = pilot_manifest.parent

    pilot = _read_json(pilot_manifest)
    digest_to_file = {
        record["observation_digest"]: record["observation_file"]
        for record in pilot.get("records", [])
    }

    annotation = _read_json(annotation_manifest)
    progressions: list[HarmonyProgression] = []
    for record in annotation.get("records", []):
        counts = record.get("target_counts", {})
        if not counts.get(HARMONIC_FUNCTION_TARGET):
            continue
        observation = _read_json(annotation_root / record["annotation_observation_file"])
        if observation.get("annotation_observation_digest") != record[
            "annotation_observation_digest"
        ]:
            raise HarmonyCorpusError(
                f"annotation observation digest mismatch for {record['target_id']}"
            )
        score_digest = record["score_observation_digest"]
        score_file = digest_to_file.get(score_digest)
        if score_file is None:
            raise HarmonyCorpusError(
                f"no pilot score observation for digest {score_digest}"
            )
        score_observation = _read_json(pilot_root / score_file)
        if score_observation.get("observation_digest") != score_digest:
            raise HarmonyCorpusError(
                f"score observation digest mismatch for {record['target_id']}"
            )
        offsets = _measure_offsets(score_observation)
        first_measure, tokens, keys = _progression_tokens(
            observation["examples"], offsets
        )
        first_measure, tokens, keys = _trim_none_edges(first_measure, tokens, keys)
        if not tokens:
            continue
        progressions.append(
            HarmonyProgression(
                score_id=record["target_id"],
                lineage_id=record["lineage_id"],
                split=record["split"],
                home_key=_home_key(keys),
                first_measure=first_measure,
                tokens=tuple(tokens),
            )
        )
    if not progressions:
        raise HarmonyCorpusError("no scores carry harmonic-function annotations")
    return progressions
