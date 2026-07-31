"""Build the concrete six-item structural-seam listening review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy.io import wavfile

from generation.composition.observation_dataset import (
    ObservationDataset,
    ScoreObservation,
)
from generation.composition.structural_context import (
    StructuralContextDatasetBuilder,
    StructuralContextError,
    StructuralContextModel,
    StructuralContextSpec,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = (
    ROOT / "experiments" / "whole_score" / "seam_review_v1" / "experiment.json"
)


class SeamReviewError(ValueError):
    """The review contract or generated package is invalid."""


@dataclass(frozen=True)
class ReviewSpec:
    experiment_id: str
    structural_context_spec: Path
    expected_structural_context_spec_digest: str
    checkpoint: Path
    expected_checkpoint_digest: str
    holdout_manifest: Path
    expected_holdout_manifest_digest: str
    output_root: Path
    item_count: int
    selection_seed: str
    qpm: float
    sample_rate: int
    lead_seconds: float
    tail_seconds: float
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "ReviewSpec":
        source = Path(path).resolve()
        try:
            value = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SeamReviewError(f"cannot read review experiment: {error}") from error
        if value.get("schema_version") != 1:
            raise SeamReviewError("review experiment schema_version must be 1")
        selection = _mapping(value.get("selection"), "selection")
        audio = _mapping(value.get("audio"), "audio")
        item_count = value.get("item_count")
        sample_rate = audio.get("sample_rate")
        if not isinstance(item_count, int) or isinstance(item_count, bool):
            raise SeamReviewError("item_count must be an integer")
        if item_count <= 0:
            raise SeamReviewError("item_count must be positive")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool):
            raise SeamReviewError("sample_rate must be an integer")
        if sample_rate < 8000:
            raise SeamReviewError("sample_rate must be at least 8000")
        return cls(
            experiment_id=_text(value.get("experiment_id"), "experiment_id"),
            structural_context_spec=_under_root(
                value.get("structural_context_spec"),
                "structural_context_spec",
            ),
            expected_structural_context_spec_digest=_digest(
                value.get("expected_structural_context_spec_digest"),
                "expected_structural_context_spec_digest",
            ),
            checkpoint=_under_root(value.get("checkpoint"), "checkpoint"),
            expected_checkpoint_digest=_digest(
                value.get("expected_checkpoint_digest"),
                "expected_checkpoint_digest",
            ),
            holdout_manifest=_under_root(
                value.get("holdout_manifest"), "holdout_manifest"
            ),
            expected_holdout_manifest_digest=_digest(
                value.get("expected_holdout_manifest_digest"),
                "expected_holdout_manifest_digest",
            ),
            output_root=_under_root(value.get("output_root"), "output_root"),
            item_count=item_count,
            selection_seed=_text(selection.get("seed"), "selection.seed"),
            qpm=_positive_number(
                audio.get("quarter_notes_per_minute"),
                "audio.quarter_notes_per_minute",
            ),
            sample_rate=sample_rate,
            lead_seconds=_nonnegative_number(
                audio.get("lead_seconds"), "audio.lead_seconds"
            ),
            tail_seconds=_nonnegative_number(
                audio.get("tail_seconds"), "audio.tail_seconds"
            ),
            raw=value,
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.raw)


@dataclass(frozen=True)
class CandidateCase:
    lineage_id: str
    target_id: str
    score_path: str
    source_digest: str
    observation: ScoreObservation
    anchor_start: int
    authentic_start: int
    nonadjacent_start: int
    model_authentic_score: float
    model_nonadjacent_score: float
    baseline_authentic_score: float
    baseline_nonadjacent_score: float

    @property
    def model_gap(self) -> float:
        return self.model_authentic_score - self.model_nonadjacent_score

    @property
    def baseline_gap(self) -> float:
        return self.baseline_authentic_score - self.baseline_nonadjacent_score


def build(spec_path: str | Path = DEFAULT_EXPERIMENT) -> Path:
    spec = ReviewSpec.load(spec_path)
    context_spec, checkpoint, dataset = _load_inputs(spec)
    cases = _select_cases(spec, context_spec, checkpoint, dataset)
    if len(cases) != spec.item_count:
        raise SeamReviewError(
            f"selected {len(cases)} cases, expected {spec.item_count}"
        )

    output = spec.output_root
    if output.exists():
        shutil.rmtree(output)
    public = output / "public"
    private = output / "private"
    audio_root = public / "audio"
    audio_root.mkdir(parents=True)
    private.mkdir(parents=True)

    public_items: list[dict[str, Any]] = []
    private_items: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        item_id = _item_id(spec.selection_seed, case)
        authentic_label = _authentic_label(spec.selection_seed, item_id)
        starts = {
            authentic_label: case.authentic_start,
            _other_label(authentic_label): case.nonadjacent_start,
        }
        rendered: dict[str, tuple[np.ndarray, float]] = {}
        for label in ("A", "B"):
            rendered[label] = render_excerpt(
                case.observation,
                anchor_start=case.anchor_start,
                continuation_start=starts[label],
                span_measures=context_spec.span_measures,
                qpm=spec.qpm,
                sample_rate=spec.sample_rate,
                lead_seconds=spec.lead_seconds,
                tail_seconds=spec.tail_seconds,
            )
        seam_seconds = rendered["A"][1]
        if not math.isclose(seam_seconds, rendered["B"][1]):
            raise SeamReviewError("blinded variants disagree on seam position")
        seam_frame = int(round(seam_seconds * spec.sample_rate))
        rendered["B"][0][:seam_frame] = rendered["A"][0][:seam_frame]
        public_audio: dict[str, dict[str, Any]] = {}
        for label in ("A", "B"):
            waveform, _ = rendered[label]
            filename = f"{item_id}_{label}.wav"
            path = audio_root / filename
            wavfile.write(path, spec.sample_rate, waveform)
            public_audio[label] = {
                "file": f"audio/{filename}",
                "digest": _file_digest(path),
                "duration_seconds": round(
                    len(waveform) / spec.sample_rate, 3
                ),
                "seam_seconds": round(seam_seconds, 6),
            }
        public_items.append(
            {
                "item_id": item_id,
                "position": position,
                "audio": public_audio,
            }
        )
        private_items.append(
            {
                "item_id": item_id,
                "lineage_id": case.lineage_id,
                "target_id": case.target_id,
                "score_path": case.score_path,
                "source_digest": case.source_digest,
                "observation_digest": case.observation.digest,
                "anchor_start_position": case.anchor_start,
                "authentic_start_position": case.authentic_start,
                "nonadjacent_start_position": case.nonadjacent_start,
                "authentic_label": authentic_label,
                "model_preferred_label": (
                    authentic_label
                    if case.model_gap > 0
                    else _other_label(authentic_label)
                ),
                "baseline_preferred_label": (
                    authentic_label
                    if case.baseline_gap > 0
                    else _other_label(authentic_label)
                ),
                "model_scores": {
                    "authentic": case.model_authentic_score,
                    "nonadjacent": case.model_nonadjacent_score,
                    "gap": case.model_gap,
                },
                "boundary_baseline_scores": {
                    "authentic": case.baseline_authentic_score,
                    "nonadjacent": case.baseline_nonadjacent_score,
                    "gap": case.baseline_gap,
                },
            }
        )

    public_manifest = {
        "schema_version": 1,
        "review_id": spec.experiment_id,
        "review_spec_digest": spec.digest,
        "prompt": "Which continuation is musically more convincing?",
        "items": public_items,
    }
    answer_key = {
        "schema_version": 1,
        "review_id": spec.experiment_id,
        "review_spec_digest": spec.digest,
        "structural_context_spec_digest": context_spec.digest,
        "checkpoint_digest": _file_digest(spec.checkpoint),
        "holdout_manifest_digest": dataset.manifest["manifest_digest"],
        "selection": (
            "one per lineage; prefer model-correct versus "
            "boundary-baseline-wrong disagreements"
        ),
        "items": private_items,
    }
    _write_json(public / "manifest.json", public_manifest)
    _write_json(private / "answer_key.json", answer_key)
    (public / "index.html").write_text(
        review_html(public_manifest), encoding="utf-8"
    )
    verify(spec_path)
    return public / "index.html"


def verify(spec_path: str | Path = DEFAULT_EXPERIMENT) -> dict[str, Any]:
    spec = ReviewSpec.load(spec_path)
    context_spec, _checkpoint, dataset = _load_inputs(spec)
    public_root = spec.output_root / "public"
    private_root = spec.output_root / "private"
    try:
        public = json.loads(
            (public_root / "manifest.json").read_text(encoding="utf-8")
        )
        private = json.loads(
            (private_root / "answer_key.json").read_text(encoding="utf-8")
        )
        html = (public_root / "index.html").read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError) as error:
        raise SeamReviewError(f"cannot verify generated review: {error}") from error
    if public.get("review_spec_digest") != spec.digest:
        raise SeamReviewError("public review spec digest does not match")
    if private.get("review_spec_digest") != spec.digest:
        raise SeamReviewError("private review spec digest does not match")
    if private.get("structural_context_spec_digest") != context_spec.digest:
        raise SeamReviewError("private structural-context digest does not match")
    if private.get("checkpoint_digest") != _file_digest(spec.checkpoint):
        raise SeamReviewError("private checkpoint digest does not match")
    if (
        private.get("holdout_manifest_digest")
        != dataset.manifest["manifest_digest"]
    ):
        raise SeamReviewError("private holdout digest does not match")
    public_items = public.get("items")
    private_items = private.get("items")
    if not isinstance(public_items, list) or len(public_items) != spec.item_count:
        raise SeamReviewError("public review has the wrong item count")
    if not isinstance(private_items, list) or len(private_items) != spec.item_count:
        raise SeamReviewError("private review has the wrong item count")
    if {item["item_id"] for item in public_items} != {
        item["item_id"] for item in private_items
    }:
        raise SeamReviewError("public and private item identities disagree")
    if len({item["lineage_id"] for item in private_items}) != spec.item_count:
        raise SeamReviewError("review does not contain one item per lineage")
    for item in public_items:
        if item["item_id"] not in html:
            raise SeamReviewError(f"HTML lacks item {item['item_id']}")
        waveforms: dict[str, np.ndarray] = {}
        for label in ("A", "B"):
            audio = item["audio"][label]
            path = public_root / audio["file"]
            if _file_digest(path) != audio["digest"]:
                raise SeamReviewError(f"audio digest mismatch: {path.name}")
            rate, waveform = wavfile.read(path)
            if rate != spec.sample_rate:
                raise SeamReviewError(f"audio sample rate mismatch: {path.name}")
            if waveform.ndim != 2 or waveform.shape[1] != 2:
                raise SeamReviewError(f"audio is not stereo: {path.name}")
            if not np.any(waveform):
                raise SeamReviewError(f"audio is silent: {path.name}")
            waveforms[label] = waveform
        seam_frame = int(
            round(item["audio"]["A"]["seam_seconds"] * spec.sample_rate)
        )
        if not np.array_equal(
            waveforms["A"][:seam_frame],
            waveforms["B"][:seam_frame],
        ):
            raise SeamReviewError(
                f"blinded variants do not share an exact anchor: "
                f"{item['item_id']}"
            )
    forbidden = {
        item["lineage_id"] for item in private_items
    } | {
        item["target_id"] for item in private_items
    } | {
        item["score_path"] for item in private_items
    }
    forbidden |= {"authentic_label", "model_scores", "baseline_scores"}
    leaked = sorted(value for value in forbidden if value and value in html)
    if leaked:
        raise SeamReviewError(f"private review data leaked into HTML: {leaked}")
    return {
        "review_id": spec.experiment_id,
        "item_count": spec.item_count,
        "lineage_count": len(private_items),
        "audio_file_count": spec.item_count * 2,
        "public_page": str(public_root / "index.html"),
        "review_spec_digest": spec.digest,
    }


def render_excerpt(
    observation: ScoreObservation,
    *,
    anchor_start: int,
    continuation_start: int,
    span_measures: int,
    qpm: float,
    sample_rate: int,
    lead_seconds: float,
    tail_seconds: float,
) -> tuple[np.ndarray, float]:
    score = _mapping(observation.data.get("score"), "score")
    measures = score.get("measures")
    events = score.get("timed_events")
    parts = score.get("parts")
    if not isinstance(measures, list) or not isinstance(events, list):
        raise SeamReviewError("observation score lacks measures or events")
    if not isinstance(parts, list):
        raise SeamReviewError("observation score lacks parts")
    anchor = measures[anchor_start : anchor_start + span_measures]
    continuation = measures[
        continuation_start : continuation_start + span_measures
    ]
    if len(anchor) != span_measures or len(continuation) != span_measures:
        raise SeamReviewError("review span falls outside the observation")
    anchor_duration = sum(
        (_rational(item["duration_ql"]) for item in anchor), Fraction()
    )
    continuation_duration = sum(
        (_rational(item["duration_ql"]) for item in continuation), Fraction()
    )
    anchor_base = _rational(anchor[0]["offset_ql"])
    continuation_base = _rational(continuation[0]["offset_ql"])
    anchor_indices = {item["index"] for item in anchor}
    continuation_indices = {item["index"] for item in continuation}
    part_names = {
        item["id"]: " ".join(
            [
                str(item.get("name", "")),
                *[
                    str(instrument.get("name", ""))
                    for instrument in item.get("instruments", [])
                ],
            ]
        )
        for item in parts
    }
    part_order = {
        item["id"]: position for position, item in enumerate(parts)
    }
    notes: list[tuple[float, float, int, str, int]] = []
    for raw_event in events:
        event = _mapping(raw_event, "timed event")
        if event.get("kind") != "note" or event.get("grace"):
            continue
        measure_index = event.get("measure_index")
        if measure_index in anchor_indices:
            relative = _rational(event["onset_ql"]) - anchor_base
        elif measure_index in continuation_indices:
            relative = (
                anchor_duration
                + _rational(event["onset_ql"])
                - continuation_base
            )
        else:
            continue
        midi = event.get("midi")
        if not isinstance(midi, int) or not 0 <= midi <= 127:
            continue
        duration = _rational(event["duration_ql"])
        segment_end = anchor_duration + continuation_duration
        duration = min(duration, segment_end - relative)
        if duration <= 0:
            continue
        part_id = str(event.get("part_id"))
        notes.append(
            (
                lead_seconds + float(relative) * 60.0 / qpm,
                float(duration) * 60.0 / qpm,
                midi,
                part_names.get(part_id, part_id),
                part_order.get(part_id, 0),
            )
        )
    total_seconds = (
        lead_seconds
        + float(anchor_duration + continuation_duration) * 60.0 / qpm
        + tail_seconds
    )
    waveform = synthesize_notes(
        notes,
        total_seconds=total_seconds,
        sample_rate=sample_rate,
        part_count=max(1, len(parts)),
    )
    seam_seconds = lead_seconds + float(anchor_duration) * 60.0 / qpm
    return waveform, seam_seconds


def synthesize_notes(
    notes: list[tuple[float, float, int, str, int]],
    *,
    total_seconds: float,
    sample_rate: int,
    part_count: int,
) -> np.ndarray:
    frames = max(1, math.ceil(total_seconds * sample_rate))
    audio = np.zeros((frames, 2), dtype=np.float64)
    base_amplitude = 0.18 / math.sqrt(max(2, part_count))
    for start, duration, midi, part_name, part_position in notes:
        first = max(0, int(round(start * sample_rate)))
        count = max(1, int(round(duration * sample_rate)))
        last = min(frames, first + count)
        if last <= first:
            continue
        time = np.arange(last - first, dtype=np.float64) / sample_rate
        frequency = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        weights = _timbre(part_name)
        phase = (midi * 0.37 + part_position * 0.19) % (2.0 * math.pi)
        wave = np.zeros_like(time)
        for harmonic, weight in enumerate(weights, start=1):
            if frequency * harmonic >= sample_rate * 0.45:
                break
            wave += weight * np.sin(
                2.0 * math.pi * frequency * harmonic * time + phase
            )
        wave /= max(1e-9, sum(abs(value) for value in weights))
        attack = min(len(time), max(1, int(sample_rate * 0.025)))
        release = min(len(time), max(1, int(sample_rate * 0.08)))
        envelope = np.ones_like(time)
        envelope[:attack] *= np.linspace(0.0, 1.0, attack, endpoint=True)
        envelope[-release:] *= np.linspace(1.0, 0.0, release, endpoint=True)
        if _family(part_name) == "string":
            envelope *= 0.94 + 0.06 * np.sin(2.0 * math.pi * 5.1 * time)
        note = base_amplitude * wave * envelope
        pan = (
            0.0
            if part_count == 1
            else -0.72 + 1.44 * part_position / (part_count - 1)
        )
        left = math.sqrt((1.0 - pan) / 2.0)
        right = math.sqrt((1.0 + pan) / 2.0)
        audio[first:last, 0] += note * left
        audio[first:last, 1] += note * right
    for delay_seconds, gain in ((0.11, 0.13), (0.19, 0.08)):
        delay = int(sample_rate * delay_seconds)
        if delay < frames:
            audio[delay:] += gain * audio[:-delay]
    audio = np.tanh(audio * 1.35)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio *= 0.92 / peak
    fade = min(frames // 2, int(sample_rate * 0.08))
    if fade:
        audio[:fade] *= np.linspace(0.0, 1.0, fade)[:, None]
        audio[-fade:] *= np.linspace(1.0, 0.0, fade)[:, None]
    return np.asarray(np.round(audio * 32767.0), dtype=np.int16)


def _select_cases(
    review_spec: ReviewSpec,
    context_spec: StructuralContextSpec,
    checkpoint: Mapping[str, Any],
    dataset: ObservationDataset,
) -> tuple[CandidateCase, ...]:
    builder = StructuralContextDatasetBuilder(context_spec, dataset)
    mean = np.asarray(checkpoint.get("feature_mean"), dtype=np.float32)
    scale = np.asarray(checkpoint.get("feature_scale"), dtype=np.float32)
    model = StructuralContextModel(context_spec, len(builder.vectorizer.feature_names))
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise SeamReviewError("checkpoint model_state must be an object")
    model.load_state_dict(state)
    model.eval()
    by_lineage: dict[str, list[CandidateCase]] = defaultdict(list)
    records = dataset.manifest.get("records")
    if not isinstance(records, list):
        raise SeamReviewError("holdout manifest records must be a list")
    for raw_record in records:
        record = _mapping(raw_record, "holdout record")
        lineage_id = _text(record.get("lineage_id"), "lineage_id")
        target_id = _text(record.get("target_id"), "target_id")
        observation = ScoreObservation.load(
            dataset.root / _text(record.get("observation_file"), "observation_file")
        )
        measures = builder.vectorizer.vectorize(observation)
        examples = builder._examples_for_score(
            target_id,
            measures,
            builder._candidate_starts(len(measures)),
            mean=mean,
            scale=scale,
        )
        for (
            anchor,
            authentic,
            nonadjacent,
            _score_id,
            anchor_start,
            authentic_start,
            nonadjacent_start,
        ) in examples:
            anchor_normalized = ((anchor - mean) / scale).astype(np.float32)
            authentic_normalized = ((authentic - mean) / scale).astype(np.float32)
            nonadjacent_normalized = (
                (nonadjacent - mean) / scale
            ).astype(np.float32)
            with torch.no_grad():
                model_authentic = float(
                    model(
                        torch.from_numpy(anchor_normalized[None, :]),
                        torch.from_numpy(authentic_normalized[None, :]),
                    )[0]
                )
                model_nonadjacent = float(
                    model(
                        torch.from_numpy(anchor_normalized[None, :]),
                        torch.from_numpy(nonadjacent_normalized[None, :]),
                    )[0]
                )
            baseline_authentic = -float(
                np.mean(
                    np.square(
                        anchor_normalized[-1] - authentic_normalized[0]
                    )
                )
            )
            baseline_nonadjacent = -float(
                np.mean(
                    np.square(
                        anchor_normalized[-1] - nonadjacent_normalized[0]
                    )
                )
            )
            by_lineage[lineage_id].append(
                CandidateCase(
                    lineage_id=lineage_id,
                    target_id=target_id,
                    score_path=_text(record.get("score_path"), "score_path"),
                    source_digest=_digest(
                        record.get("source_digest"), "source_digest"
                    ),
                    observation=observation,
                    anchor_start=int(anchor_start),
                    authentic_start=int(authentic_start),
                    nonadjacent_start=int(nonadjacent_start),
                    model_authentic_score=model_authentic,
                    model_nonadjacent_score=model_nonadjacent,
                    baseline_authentic_score=baseline_authentic,
                    baseline_nonadjacent_score=baseline_nonadjacent,
                )
            )
    selected: list[CandidateCase] = []
    for lineage_id, cases in sorted(by_lineage.items()):
        disagreements = [
            case
            for case in cases
            if case.model_gap > 0.0 and case.baseline_gap <= 0.0
        ]
        model_correct = [case for case in cases if case.model_gap > 0.0]
        pool = disagreements or model_correct or cases
        selected.append(
            max(
                pool,
                key=lambda case: (
                    case.model_gap - case.baseline_gap,
                    _stable_rank(
                        review_spec.selection_seed,
                        lineage_id,
                        case.target_id,
                        case.anchor_start,
                    ),
                ),
            )
        )
    if len(selected) != review_spec.item_count:
        raise SeamReviewError(
            f"holdout has {len(selected)} lineages, expected "
            f"{review_spec.item_count}"
        )
    selected.sort(
        key=lambda case: _stable_rank(
            review_spec.selection_seed, "display", case.lineage_id
        )
    )
    return tuple(selected)


def _load_inputs(
    spec: ReviewSpec,
) -> tuple[StructuralContextSpec, Mapping[str, Any], ObservationDataset]:
    context_spec = StructuralContextSpec.load(spec.structural_context_spec)
    if context_spec.digest != spec.expected_structural_context_spec_digest:
        raise SeamReviewError("structural-context spec digest does not match")
    if _file_digest(spec.checkpoint) != spec.expected_checkpoint_digest:
        raise SeamReviewError("structural-context checkpoint digest does not match")
    try:
        checkpoint = torch.load(
            spec.checkpoint, map_location="cpu", weights_only=False
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise SeamReviewError(f"cannot load checkpoint: {error}") from error
    if not isinstance(checkpoint, Mapping):
        raise SeamReviewError("checkpoint must be an object")
    if checkpoint.get("experiment_spec_digest") != context_spec.digest:
        raise SeamReviewError("checkpoint and structural-context spec disagree")
    dataset = ObservationDataset.load(spec.holdout_manifest)
    if (
        dataset.manifest.get("manifest_digest")
        != spec.expected_holdout_manifest_digest
    ):
        raise SeamReviewError("holdout manifest digest does not match")
    return context_spec, checkpoint, dataset


def review_html(manifest: Mapping[str, Any]) -> str:
    public_json = json.dumps(
        manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    return _HTML.replace("__PUBLIC_MANIFEST__", public_json)


def _family(name: str) -> str:
    lowered = name.lower()
    if any(word in lowered for word in ("violin", "viola", "cello", "bass")):
        return "string"
    if any(word in lowered for word in ("horn", "trumpet", "trombone", "tuba")):
        return "brass"
    if any(word in lowered for word in ("flute", "oboe", "clarinet", "bassoon")):
        return "woodwind"
    if any(word in lowered for word in ("harp", "piano", "celesta", "organ")):
        return "keyboard"
    return "neutral"


def _timbre(name: str) -> tuple[float, ...]:
    family = _family(name)
    if family == "string":
        return (1.0, 0.42, 0.22, 0.13, 0.08)
    if family == "brass":
        return (1.0, 0.62, 0.36, 0.18, 0.1)
    if family == "woodwind":
        return (1.0, 0.22, 0.3, 0.08, 0.12)
    if family == "keyboard":
        return (1.0, 0.31, 0.13, 0.06)
    return (1.0, 0.27, 0.12)


def _item_id(seed: str, case: CandidateCase) -> str:
    digest = hashlib.sha256(
        (
            f"{seed}\0{case.lineage_id}\0{case.target_id}\0"
            f"{case.anchor_start}\0{case.nonadjacent_start}"
        ).encode("utf-8")
    ).hexdigest()
    return f"seam-{digest[:10]}"


def _authentic_label(seed: str, item_id: str) -> str:
    return "A" if _stable_rank(seed, item_id, "blind") % 2 == 0 else "B"


def _other_label(label: str) -> str:
    return "B" if label == "A" else "A"


def _stable_rank(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SeamReviewError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SeamReviewError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if (
        len(text) != 71
        or not text.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise SeamReviewError(f"{label} must be a SHA-256 digest")
    return text


def _positive_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise SeamReviewError(f"{label} must be a positive number")
    return float(value)


def _nonnegative_number(value: object, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise SeamReviewError(f"{label} must be a non-negative number")
    return float(value)


def _under_root(value: object, label: str) -> Path:
    relative = Path(_text(value, label))
    if relative.is_absolute() or ".." in relative.parts:
        raise SeamReviewError(f"{label} must stay under the repository root")
    return ROOT / relative


def _rational(value: object) -> Fraction:
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise SeamReviewError(f"invalid rational value: {value!r}") from error


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
    except OSError as error:
        raise SeamReviewError(f"cannot read {path}: {error}") from error
    return f"sha256:{hasher.hexdigest()}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Structural seam review</title>
  <style>
    :root {
      color-scheme: dark;
      --ink: #f6f0e4;
      --muted: #b9b0a1;
      --panel: #17191d;
      --line: #353942;
      --accent: #e8b45e;
      --accent-2: #77b8a7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 15%, #2b2931 0, transparent 35%),
        radial-gradient(circle at 90% 85%, #17332f 0, transparent 32%),
        #0d0f12;
      font: 17px/1.45 ui-sans-serif, system-ui, sans-serif;
    }
    main { width: min(920px, calc(100% - 32px)); margin: 0 auto; padding: 34px 0 60px; }
    h1 { margin: 0; font: 600 clamp(28px, 5vw, 48px)/1.05 Georgia, serif; }
    .lede { color: var(--muted); max-width: 680px; margin: 12px 0 24px; }
    .progress { height: 7px; background: #24272e; border-radius: 99px; overflow: hidden; }
    .progress > div { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent-2)); transition: width .25s; }
    .meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 14px; margin: 9px 2px 24px; }
    .card { background: color-mix(in srgb, var(--panel) 94%, transparent); border: 1px solid var(--line); border-radius: 18px; padding: clamp(18px, 4vw, 34px); box-shadow: 0 22px 70px #0007; }
    .question { font: 600 24px/1.2 Georgia, serif; margin: 0 0 8px; }
    .hint { color: var(--muted); margin: 0 0 22px; }
    .players { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }
    .variant { border: 1px solid var(--line); border-radius: 14px; padding: 18px; background: #111318; }
    .variant h2 { margin: 0 0 12px; font-size: 30px; }
    audio { width: 100%; margin-bottom: 12px; }
    button { font: inherit; color: inherit; cursor: pointer; }
    .play { width: 100%; border: 1px solid #4b515c; border-radius: 9px; background: #242932; padding: 10px; }
    .choices { display: grid; grid-template-columns: 1fr .75fr 1fr; gap: 12px; margin-top: 24px; }
    .choice { border: 1px solid #555c68; border-radius: 12px; background: #20242b; padding: 15px 8px; font-weight: 700; }
    .choice:hover, .choice.selected { color: #15110a; border-color: var(--accent); background: var(--accent); }
    .nav { display: flex; justify-content: space-between; gap: 12px; margin-top: 18px; }
    .nav button { border: 0; background: transparent; color: var(--muted); padding: 8px 0; }
    .finish { display: none; }
    .finish textarea { width: 100%; min-height: 120px; margin: 14px 0; background: #0b0d10; color: var(--ink); border: 1px solid var(--line); border-radius: 10px; padding: 12px; }
    .primary { border: 0; border-radius: 11px; background: var(--accent-2); color: #07110e; padding: 13px 20px; font-weight: 800; }
    .status { color: var(--accent-2); min-height: 24px; margin-top: 10px; }
    @media (max-width: 700px) {
      .players { grid-template-columns: 1fr; }
      .choices { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <h1>Which continuation works better?</h1>
  <p class="lede">Six short, anonymous comparisons. Both versions share the same opening. Listen across the seam, then choose A, B, or Same. No written explanation is needed.</p>
  <div class="progress"><div id="bar"></div></div>
  <div class="meta"><span id="count"></span><span id="answered"></span></div>
  <section class="card" id="review">
    <p class="question">Which continuation is musically more convincing?</p>
    <p class="hint" id="hint"></p>
    <div class="players">
      <article class="variant"><h2>A</h2><audio id="audioA" controls preload="metadata"></audio><button class="play" data-label="A">Play A from the seam</button></article>
      <article class="variant"><h2>B</h2><audio id="audioB" controls preload="metadata"></audio><button class="play" data-label="B">Play B from the seam</button></article>
    </div>
    <div class="choices">
      <button class="choice" data-choice="A">A is better</button>
      <button class="choice" data-choice="S">Same</button>
      <button class="choice" data-choice="B">B is better</button>
    </div>
    <div class="nav"><button id="back">← Previous</button><button id="next">Next →</button></div>
  </section>
  <section class="card finish" id="finish">
    <p class="question">Review complete.</p>
    <p class="hint">Click once to copy the result, then paste it into the chat. That is the only handoff needed.</p>
    <textarea id="result" readonly></textarea>
    <button class="primary" id="copy">Copy results</button>
    <button class="choice" id="download">Download backup</button>
    <div class="status" id="status"></div>
  </section>
</main>
<script>
const manifest = __PUBLIC_MANIFEST__;
const storageKey = manifest.review_id + ":" + manifest.review_spec_digest;
let choices = JSON.parse(localStorage.getItem(storageKey) || "{}");
let index = manifest.items.findIndex(item => !choices[item.item_id]);
if (index < 0) index = manifest.items.length;
const $ = id => document.getElementById(id);
const audios = {A: $("audioA"), B: $("audioB")};

function stopAudio() {
  Object.values(audios).forEach(audio => audio.pause());
}
function save() {
  localStorage.setItem(storageKey, JSON.stringify(choices));
}
function resultValue() {
  return JSON.stringify({
    schema_version: 1,
    review_id: manifest.review_id,
    review_spec_digest: manifest.review_spec_digest,
    choices: manifest.items.map(item => ({
      item_id: item.item_id,
      choice: choices[item.item_id]
    }))
  });
}
function render() {
  stopAudio();
  const done = manifest.items.every(item => choices[item.item_id]);
  $("review").style.display = done && index >= manifest.items.length ? "none" : "block";
  $("finish").style.display = done && index >= manifest.items.length ? "block" : "none";
  const answered = manifest.items.filter(item => choices[item.item_id]).length;
  $("bar").style.width = `${100 * answered / manifest.items.length}%`;
  $("answered").textContent = `${answered} answered`;
  if (done && index >= manifest.items.length) {
    $("count").textContent = "Complete";
    $("result").value = resultValue();
    return;
  }
  if (index >= manifest.items.length) index = manifest.items.length - 1;
  const item = manifest.items[index];
  $("count").textContent = `Comparison ${index + 1} of ${manifest.items.length}`;
  $("hint").textContent = `The shared opening ends at ${item.audio.A.seam_seconds.toFixed(1)} seconds.`;
  for (const label of ["A", "B"]) {
    audios[label].src = item.audio[label].file;
  }
  document.querySelectorAll("[data-choice]").forEach(button => {
    button.classList.toggle("selected", button.dataset.choice === choices[item.item_id]);
  });
  $("back").disabled = index === 0;
}
document.querySelectorAll(".play").forEach(button => {
  button.addEventListener("click", () => {
    const label = button.dataset.label;
    const item = manifest.items[index];
    stopAudio();
    audios[label].currentTime = Math.max(0, item.audio[label].seam_seconds - 0.7);
    audios[label].play();
  });
});
document.querySelectorAll("[data-choice]").forEach(button => {
  button.addEventListener("click", () => {
    choices[manifest.items[index].item_id] = button.dataset.choice;
    save();
    window.setTimeout(() => { index += 1; render(); }, 180);
  });
});
$("back").addEventListener("click", () => { if (index > 0) { index -= 1; render(); } });
$("next").addEventListener("click", () => { if (index < manifest.items.length) { index += 1; render(); } });
$("copy").addEventListener("click", () => {
  $("result").select();
  const copied = document.execCommand("copy");
  $("status").textContent = copied ? "Copied. Paste it into the chat." : "Select the result text and copy it.";
});
$("download").addEventListener("click", () => {
  const blob = new Blob([resultValue() + "\\n"], {type: "application/json"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "seam-review-v1-results.json";
  link.click();
  URL.revokeObjectURL(link.href);
});
document.addEventListener("keydown", event => {
  const choice = event.key.toUpperCase();
  if (["A", "B", "S"].includes(choice) && $("review").style.display !== "none") {
    document.querySelector(`[data-choice="${choice}"]`).click();
  }
});
render();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--spec", default=str(DEFAULT_EXPERIMENT))
    arguments = parser.parse_args(argv)
    try:
        result: object
        if arguments.command == "build":
            result = {"public_page": str(build(arguments.spec))}
        else:
            result = verify(arguments.spec)
    except (SeamReviewError, StructuralContextError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
