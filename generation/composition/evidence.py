"""Read-only ML views over Ruby-owned composition evidence.

Partitura creates, validates, replays, and stores every transition and review.
This module only validates immutable dataset records and joins them for learned
critics and policies. It does not parse scores or apply source patches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

TRAJECTORY_SCHEMA_VERSION = 2
REVIEW_SCHEMA_VERSION = 1
PREFERENCE_SCHEMA_VERSION = 1
ORIGINAL_CANDIDATE_ID = "original"

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ORIGINS = {"deterministic": "unrated", "agent": "medium"}
_REVIEW_SCALES = {"local", "seam", "section", "global", "export"}
_OUTCOMES = {"a", "b", "tie", "abstain"}
_PURPOSES = {"training", "held_out_evaluation"}


class EvidenceError(ValueError):
    """A Ruby evidence record is malformed or refers to absent evidence."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{label} must be a non-empty string")
    return value


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _digest(value: object, label: str) -> str:
    text = _text(value, label)
    if not _DIGEST.fullmatch(text):
        raise EvidenceError(f"{label} must be a sha256 digest")
    return text


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise EvidenceError(f"{label} must be an array")
    return tuple(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {_text(key, "object key"): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise EvidenceError(f"unsupported evidence value {type(value).__name__}")


def _records(path: str | Path, label: str) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                records.append(_mapping(parsed, label))
            except (json.JSONDecodeError, EvidenceError) as error:
                raise EvidenceError(
                    f"invalid {label} record at line {line_number}: {error}"
                ) from error
    return tuple(records)


@dataclass(frozen=True)
class TrajectoryRecord:
    """One self-contained Partitura transition, including rejected candidates."""

    transition_id: str
    before_snapshot: Mapping[str, Any]
    before_source: str
    before_source_digest: str
    trajectory_context: Mapping[str, Any]
    action: Mapping[str, Any]
    candidates: tuple[Mapping[str, Any], ...]
    decision: str
    after_graph_digest: str
    after_snapshot_digest: str
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TrajectoryRecord:
        if value.get("schema_version") != TRAJECTORY_SCHEMA_VERSION:
            raise EvidenceError("unsupported trajectory schema")
        before_snapshot = _mapping(value.get("before_snapshot"), "before_snapshot")
        before_source = _text(value.get("before_source"), "before_source")
        source_digest = _digest(
            value.get("before_source_digest"), "before_source_digest"
        )
        if _sha256(before_source) != source_digest:
            raise EvidenceError("before_source does not match before_source_digest")
        if before_snapshot.get("graph_digest") != value.get("before_graph_digest"):
            raise EvidenceError(
                "before_snapshot graph digest does not match transition"
            )
        if before_snapshot.get("snapshot_digest") != value.get(
            "before_snapshot_digest"
        ):
            raise EvidenceError("before_snapshot digest does not match transition")

        context = _mapping(value.get("trajectory_context"), "trajectory_context")
        origin = _text(context.get("origin"), "trajectory origin")
        quality = _text(context.get("quality_label"), "trajectory quality_label")
        if _ORIGINS.get(origin) != quality:
            raise EvidenceError(f"{origin} trajectory has invalid quality label")
        _text(context.get("run_id"), "trajectory run_id")

        candidates = tuple(
            _mapping(item, "candidate assessment")
            for item in _sequence(value.get("candidates"), "candidates")
        )
        cls._validate_candidate_ids(candidates)
        decision = _text(value.get("decision"), "decision")
        if decision not in {"accept", "keep_original", "backtrack", "defer"}:
            raise EvidenceError(f"unsupported transition decision {decision!r}")
        action = _mapping(value.get("action"), "action")
        if action.get("base_snapshot_digest") != before_snapshot.get("snapshot_digest"):
            raise EvidenceError("action does not match before_snapshot")

        return cls(
            transition_id=_text(value.get("transition_id"), "transition_id"),
            before_snapshot=_freeze(before_snapshot),
            before_source=before_source,
            before_source_digest=source_digest,
            trajectory_context=_freeze(context),
            action=_freeze(action),
            candidates=tuple(_freeze(item) for item in candidates),
            decision=decision,
            after_graph_digest=_digest(
                value.get("after_graph_digest"), "after_graph_digest"
            ),
            after_snapshot_digest=_digest(
                value.get("after_snapshot_digest"), "after_snapshot_digest"
            ),
            raw=_freeze(value),
        )

    @staticmethod
    def _validate_candidate_ids(candidates: tuple[Mapping[str, Any], ...]) -> None:
        ids = []
        for assessment in candidates:
            candidate = _mapping(assessment.get("candidate"), "candidate")
            ids.append(_text(candidate.get("candidate_id"), "candidate_id"))
            _digest(candidate.get("patch_digest"), "candidate patch_digest")
            if not candidate.get("source_patch") and not candidate.get("artifact"):
                raise EvidenceError(
                    "candidate evidence lacks its replayable patch payload"
                )
        if len(ids) != len(set(ids)):
            raise EvidenceError("transition candidate ids must be unique")

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(
            _mapping(item["candidate"], "candidate")["candidate_id"]
            for item in self.candidates
        )

    @property
    def origin(self) -> str:
        return self.trajectory_context["origin"]

    @property
    def run_id(self) -> str:
        return self.trajectory_context["run_id"]

    @property
    def quality_label(self) -> str:
        return self.trajectory_context["quality_label"]


@dataclass(frozen=True)
class PairwiseReviewRecord:
    review_id: str
    transition_id: str
    scale: str
    variants: Mapping[str, str]
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PairwiseReviewRecord:
        if value.get("schema_version") != REVIEW_SCHEMA_VERSION:
            raise EvidenceError("unsupported review schema")
        if value.get("kind") != "pairwise_review" or value.get("blind") is not True:
            raise EvidenceError("review must be a private blinded pairwise record")
        scale = _text(value.get("scale"), "review scale")
        if scale not in _REVIEW_SCALES:
            raise EvidenceError(f"unsupported review scale {scale!r}")
        variants = {}
        for item in _sequence(value.get("variants"), "review variants"):
            variant = _mapping(item, "review variant")
            label = _text(variant.get("label"), "review label")
            candidate_id = _text(variant.get("candidate_id"), "review candidate_id")
            variants[label] = candidate_id
        if set(variants) != {"A", "B"} or len(set(variants.values())) != 2:
            raise EvidenceError("review requires distinct A and B candidates")
        return cls(
            review_id=_text(value.get("review_id"), "review_id"),
            transition_id=_text(value.get("transition_id"), "transition_id"),
            scale=scale,
            variants=_freeze(variants),
            raw=_freeze(value),
        )


@dataclass(frozen=True)
class HumanPreferenceRecord:
    preference_id: str
    review_id: str
    transition_id: str
    outcome: str
    preferred_candidate_id: str | None
    other_candidate_id: str | None
    scale: str
    purpose: str
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HumanPreferenceRecord:
        if value.get("schema_version") != PREFERENCE_SCHEMA_VERSION:
            raise EvidenceError("unsupported preference schema")
        if value.get("kind") != "human_preference" or value.get("blind") is not True:
            raise EvidenceError("preference must be a blinded human record")
        outcome = _text(value.get("outcome"), "preference outcome")
        purpose = _text(value.get("purpose"), "preference purpose")
        scale = _text(value.get("scale"), "preference scale")
        if outcome not in _OUTCOMES:
            raise EvidenceError(f"unsupported preference outcome {outcome!r}")
        if purpose not in _PURPOSES:
            raise EvidenceError(f"unsupported preference purpose {purpose!r}")
        if scale not in _REVIEW_SCALES:
            raise EvidenceError(f"unsupported preference scale {scale!r}")
        preferred = value.get("preferred_candidate_id")
        other = value.get("other_candidate_id")
        if outcome in {"a", "b"}:
            preferred = _text(preferred, "preferred_candidate_id")
            other = _text(other, "other_candidate_id")
            if preferred == other:
                raise EvidenceError("preference candidates must be distinct")
        elif preferred is not None or other is not None:
            raise EvidenceError("tie and abstain records may not resolve candidates")
        return cls(
            preference_id=_text(value.get("preference_id"), "preference_id"),
            review_id=_text(value.get("review_id"), "preference review_id"),
            transition_id=_text(value.get("transition_id"), "preference transition_id"),
            outcome=outcome,
            preferred_candidate_id=preferred,
            other_candidate_id=other,
            scale=scale,
            purpose=purpose,
            raw=_freeze(value),
        )


@dataclass(frozen=True)
class PairwiseExample:
    """A resolved preference joined to its full Ruby trajectory evidence."""

    transition: TrajectoryRecord
    review: PairwiseReviewRecord
    preference: HumanPreferenceRecord

    @property
    def preferred_candidate_id(self) -> str:
        value = self.preference.preferred_candidate_id
        if value is None:
            raise EvidenceError("unresolved preference cannot form a pairwise example")
        return value

    @property
    def other_candidate_id(self) -> str:
        value = self.preference.other_candidate_id
        if value is None:
            raise EvidenceError("unresolved preference cannot form a pairwise example")
        return value


@dataclass(frozen=True)
class CompositionDataset:
    trajectories: tuple[TrajectoryRecord, ...]
    reviews: tuple[PairwiseReviewRecord, ...]
    preferences: tuple[HumanPreferenceRecord, ...]

    def __post_init__(self) -> None:
        transitions = self._unique(self.trajectories, "transition_id", "transition")
        reviews = self._unique(self.reviews, "review_id", "review")
        self._unique(self.preferences, "preference_id", "preference")
        self._unique(self.preferences, "review_id", "preference review")
        self._validate_trajectory()
        for review in self.reviews:
            transition = transitions.get(review.transition_id)
            if transition is None:
                raise EvidenceError(f"review {review.review_id} has no transition")
            allowed = set(transition.candidate_ids) | {ORIGINAL_CANDIDATE_ID}
            if not set(review.variants.values()).issubset(allowed):
                raise EvidenceError(
                    f"review {review.review_id} names an absent candidate"
                )
        for preference in self.preferences:
            review = reviews.get(preference.review_id)
            self._validate_preference_join(preference, review)

    @classmethod
    def from_jsonl(
        cls,
        trajectory_path: str | Path,
        review_path: str | Path,
        preference_path: str | Path,
    ) -> CompositionDataset:
        return cls(
            trajectories=tuple(
                TrajectoryRecord.from_dict(item)
                for item in _records(trajectory_path, "trajectory")
            ),
            reviews=tuple(
                PairwiseReviewRecord.from_dict(item)
                for item in _records(review_path, "review")
            ),
            preferences=tuple(
                HumanPreferenceRecord.from_dict(item)
                for item in _records(preference_path, "preference")
            ),
        )

    def training_pairs(self) -> tuple[PairwiseExample, ...]:
        return self._pairs("training")

    def held_out_pairs(self) -> tuple[PairwiseExample, ...]:
        return self._pairs("held_out_evaluation")

    @staticmethod
    def _unique(records: tuple[Any, ...], attribute: str, label: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for record in records:
            record_id = getattr(record, attribute)
            if record_id in indexed:
                raise EvidenceError(f"duplicate {label} id {record_id}")
            indexed[record_id] = record
        return indexed

    def _validate_trajectory(self) -> None:
        contexts = {
            (item.run_id, item.origin, item.quality_label) for item in self.trajectories
        }
        if len(contexts) > 1:
            raise EvidenceError("trajectory file mixes incompatible run contexts")
        for before, after in zip(self.trajectories, self.trajectories[1:]):
            if before.after_graph_digest != after.before_snapshot.get(
                "graph_digest"
            ) or before.after_snapshot_digest != after.before_snapshot.get(
                "snapshot_digest"
            ):
                raise EvidenceError("trajectory records are not contiguous")

    @staticmethod
    def _validate_preference_join(
        preference: HumanPreferenceRecord,
        review: PairwiseReviewRecord | None,
    ) -> None:
        if review is None:
            raise EvidenceError(f"preference {preference.preference_id} has no review")
        if (
            preference.transition_id != review.transition_id
            or preference.scale != review.scale
        ):
            raise EvidenceError(
                f"preference {preference.preference_id} disagrees with its review"
            )
        if preference.outcome not in {"a", "b"}:
            return
        chosen = "A" if preference.outcome == "a" else "B"
        other = "B" if chosen == "A" else "A"
        if (
            preference.preferred_candidate_id != review.variants[chosen]
            or preference.other_candidate_id != review.variants[other]
        ):
            raise EvidenceError(
                f"preference {preference.preference_id} reveals the wrong blind mapping"
            )

    def _pairs(self, purpose: str) -> tuple[PairwiseExample, ...]:
        transitions = {item.transition_id: item for item in self.trajectories}
        reviews = {item.review_id: item for item in self.reviews}
        return tuple(
            PairwiseExample(
                transition=transitions[item.transition_id],
                review=reviews[item.review_id],
                preference=item,
            )
            for item in self.preferences
            if item.purpose == purpose and item.outcome in {"a", "b"}
        )
