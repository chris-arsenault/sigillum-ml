"""Versioned data-transfer objects for Partitura's ML protocol.

This module deliberately contains no score model, scheduler, patch executor,
promotion logic, or trajectory store. Ruby Partitura owns those operations.
Python receives immutable observations and returns model proposals, learned
critic results, and selections.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
ORIGINAL_CANDIDATE_ID = "original"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")

JsonValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)


class ProtocolError(ValueError):
    """The message is not a valid Partitura ML protocol record."""


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolError(f"{label} must be a non-empty string")
    return value


def _digest(value: object, label: str) -> str:
    text = _required_text(value, label)
    if not _DIGEST.fullmatch(text):
        raise ProtocolError(f"{label} must be a sha256 digest")
    return text


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ProtocolError(f"{label} must be an array")
    return value


def _freeze(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                _required_text(key, "JSON object key"): _freeze(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ProtocolError(f"unsupported JSON value {type(value).__name__}")


def _thaw(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _message(value: Mapping[str, Any], kind: str) -> Mapping[str, Any]:
    try:
        schema_version = value["schema_version"]
        actual_kind = value["kind"]
    except KeyError as error:
        raise ProtocolError(f"protocol message lacks {error.args[0]}") from error
    if schema_version != SCHEMA_VERSION:
        raise ProtocolError(f"unsupported protocol schema {schema_version!r}")
    if actual_kind != kind:
        raise ProtocolError(f"expected {kind}, got {actual_kind!r}")
    return value


def _load_json(value: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ProtocolError(f"invalid protocol JSON: {error}") from error
    return _object(parsed, "protocol message")


class _JsonDTO:
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True)
class ProposalRequest(_JsonDTO):
    """Opaque score observation and one Ruby-scheduled composition action."""

    request_id: str
    snapshot: Mapping[str, JsonValue]
    action: Mapping[str, JsonValue]
    source_name: str
    source_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "request_id", _required_text(self.request_id, "request_id")
        )
        object.__setattr__(
            self, "snapshot", _freeze(_object(self.snapshot, "snapshot"))
        )
        object.__setattr__(self, "action", _freeze(_object(self.action, "action")))
        object.__setattr__(
            self, "source_name", _required_text(self.source_name, "source_name")
        )
        object.__setattr__(
            self, "source_digest", _digest(self.source_digest, "source_digest")
        )
        if self.snapshot.get("snapshot_digest") != self.action.get(
            "base_snapshot_digest"
        ):
            raise ProtocolError("proposal snapshot and action digests do not match")
        if self.snapshot.get("graph_digest") != self.action.get("base_graph_digest"):
            raise ProtocolError("proposal graph and action digests do not match")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProposalRequest":
        data = _message(value, "proposal_request")
        try:
            return cls(
                request_id=data["request_id"],
                snapshot=data["snapshot"],
                action=data["action"],
                source_name=data["source_name"],
                source_digest=data["source_digest"],
            )
        except KeyError as error:
            raise ProtocolError(f"proposal request lacks {error.args[0]}") from error

    @classmethod
    def from_json(cls, value: str) -> "ProposalRequest":
        return cls.from_dict(_load_json(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "proposal_request",
            "request_id": self.request_id,
            "snapshot": _thaw(self.snapshot),
            "action": _thaw(self.action),
            "source_name": self.source_name,
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class CandidateProposal(_JsonDTO):
    """An ML-produced source-patch record bound to one Ruby action."""

    candidate_id: str
    base_snapshot_digest: str
    target_path: str
    lens: str
    operator: str
    patch_digest: str
    touched_paths: tuple[str, ...]
    description: str
    source_patch: str | None = None
    artifact: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _digest(self.base_snapshot_digest, "base_snapshot_digest")
        _required_text(self.target_path, "target_path")
        _required_text(self.lens, "lens")
        _required_text(self.operator, "operator")
        _digest(self.patch_digest, "patch_digest")
        _required_text(self.description, "description")
        if not self.touched_paths or len(set(self.touched_paths)) != len(
            self.touched_paths
        ):
            raise ProtocolError("touched_paths must be non-empty and unique")
        if self.target_path not in self.touched_paths:
            raise ProtocolError("touched_paths must include target_path")
        if (self.source_patch is None) == (self.artifact is None):
            raise ProtocolError("candidate needs exactly one patch payload")
        if (
            self.source_patch is not None
            and _sha256(self.source_patch) != self.patch_digest
        ):
            raise ProtocolError("source_patch does not match patch_digest")

    @classmethod
    def inline(
        cls,
        request: ProposalRequest,
        *,
        source_patch: str,
        description: str,
        touched_paths: Sequence[str] | None = None,
        candidate_id: str | None = None,
    ) -> "CandidateProposal":
        patch_digest = _sha256(source_patch)
        target = _required_text(request.action.get("target_path"), "action target_path")
        return cls(
            candidate_id=candidate_id
            or f"candidate:{patch_digest.split(':', 1)[1][:20]}",
            base_snapshot_digest=_digest(
                request.action.get("base_snapshot_digest"),
                "action base_snapshot_digest",
            ),
            target_path=target,
            lens=_required_text(request.action.get("lens"), "action lens"),
            operator=_required_text(request.action.get("operator"), "action operator"),
            patch_digest=patch_digest,
            touched_paths=tuple(touched_paths or (target,)),
            description=description,
            source_patch=source_patch,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateProposal":
        try:
            paths = _array(value["touched_paths"], "touched_paths")
            return cls(
                candidate_id=value["candidate_id"],
                base_snapshot_digest=value["base_snapshot_digest"],
                target_path=value["target_path"],
                lens=value["lens"],
                operator=value["operator"],
                patch_digest=value["patch_digest"],
                touched_paths=tuple(
                    _required_text(path, "touched path") for path in paths
                ),
                description=value["description"],
                source_patch=value.get("source_patch"),
                artifact=value.get("artifact"),
            )
        except KeyError as error:
            raise ProtocolError(f"candidate lacks {error.args[0]}") from error

    def validate_for(self, request: ProposalRequest) -> None:
        expected = (
            request.action.get("base_snapshot_digest"),
            request.action.get("target_path"),
            request.action.get("lens"),
            request.action.get("operator"),
        )
        actual = (
            self.base_snapshot_digest,
            self.target_path,
            self.lens,
            self.operator,
        )
        if actual != expected:
            raise ProtocolError("candidate does not implement the requested action")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "candidate_id": self.candidate_id,
            "base_snapshot_digest": self.base_snapshot_digest,
            "target_path": self.target_path,
            "lens": self.lens,
            "operator": self.operator,
            "patch_digest": self.patch_digest,
            "touched_paths": list(self.touched_paths),
            "description": self.description,
        }
        if self.source_patch is not None:
            value["source_patch"] = self.source_patch
        if self.artifact is not None:
            value["artifact"] = self.artifact
        return value


@dataclass(frozen=True)
class ProposalResponse(_JsonDTO):
    request_id: str
    producer: str
    candidates: tuple[CandidateProposal, ...]

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.producer, "producer")
        ids = [candidate.candidate_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ProtocolError("candidate ids must be unique")

    @classmethod
    def create(
        cls,
        request: ProposalRequest,
        *,
        producer: str,
        candidates: Sequence[CandidateProposal],
    ) -> "ProposalResponse":
        response = cls(request.request_id, producer, tuple(candidates))
        response.validate_for(request)
        return response

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProposalResponse":
        data = _message(value, "proposal_response")
        try:
            return cls(
                request_id=data["request_id"],
                producer=data["producer"],
                candidates=tuple(
                    CandidateProposal.from_dict(item)
                    for item in _array(data["candidates"], "candidates")
                ),
            )
        except KeyError as error:
            raise ProtocolError(f"proposal response lacks {error.args[0]}") from error

    @classmethod
    def from_json(cls, value: str) -> "ProposalResponse":
        return cls.from_dict(_load_json(value))

    def validate_for(self, request: ProposalRequest) -> None:
        if self.request_id != request.request_id:
            raise ProtocolError("proposal response belongs to another request")
        for candidate in self.candidates:
            candidate.validate_for(request)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "proposal_response",
            "request_id": self.request_id,
            "producer": self.producer,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


@dataclass(frozen=True)
class SelectionRequest(_JsonDTO):
    """Ruby-validated candidate evidence offered to learned critics/policies."""

    request_id: str
    proposal_request_id: str
    snapshot: Mapping[str, JsonValue]
    action: Mapping[str, JsonValue]
    original_candidate_id: str
    assessments: tuple[Mapping[str, JsonValue], ...]
    candidate_observations: Mapping[str, JsonValue] = MappingProxyType({})

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.proposal_request_id, "proposal_request_id")
        object.__setattr__(
            self, "snapshot", _freeze(_object(self.snapshot, "snapshot"))
        )
        object.__setattr__(self, "action", _freeze(_object(self.action, "action")))
        object.__setattr__(
            self,
            "assessments",
            tuple(_freeze(_object(item, "assessment")) for item in self.assessments),
        )
        object.__setattr__(
            self,
            "candidate_observations",
            _freeze(
                _object(self.candidate_observations, "candidate_observations")
            ),
        )
        if self.original_candidate_id != ORIGINAL_CANDIDATE_ID:
            raise ProtocolError("selection request does not expose the original")
        if not self.assessments:
            raise ProtocolError("selection request needs at least one assessment")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ProtocolError("selection candidate ids must be unique")
        unknown_observations = (
            set(self.candidate_observations) - set(self.candidate_ids)
        )
        if unknown_observations:
            raise ProtocolError(
                "selection observations name unknown candidates: "
                + ", ".join(sorted(unknown_observations))
            )
        for candidate_id, raw_observation in self.candidate_observations.items():
            observation = _object(
                raw_observation,
                f"candidate observation for {candidate_id}",
            )
            if observation.get("schema_version") != SCHEMA_VERSION:
                raise ProtocolError(
                    f"candidate observation for {candidate_id} has "
                    "unsupported schema"
                )
            _digest(
                observation.get("observation_digest"),
                f"candidate observation digest for {candidate_id}",
            )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        try:
            return tuple(
                _required_text(
                    _object(assessment["candidate"], "candidate")["candidate_id"],
                    "candidate_id",
                )
                for assessment in self.assessments
            )
        except KeyError as error:
            raise ProtocolError(f"assessment lacks {error.args[0]}") from error

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionRequest":
        data = _message(value, "selection_request")
        try:
            return cls(
                request_id=data["request_id"],
                proposal_request_id=data["proposal_request_id"],
                snapshot=data["snapshot"],
                action=data["action"],
                original_candidate_id=data["original_candidate_id"],
                assessments=tuple(_array(data["assessments"], "assessments")),
                candidate_observations=data.get("candidate_observations", {}),
            )
        except KeyError as error:
            raise ProtocolError(f"selection request lacks {error.args[0]}") from error

    @classmethod
    def from_json(cls, value: str) -> "SelectionRequest":
        return cls.from_dict(_load_json(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "selection_request",
            "request_id": self.request_id,
            "proposal_request_id": self.proposal_request_id,
            "snapshot": _thaw(self.snapshot),
            "action": _thaw(self.action),
            "original_candidate_id": self.original_candidate_id,
            "assessments": [_thaw(item) for item in self.assessments],
            "candidate_observations": _thaw(self.candidate_observations),
        }


@dataclass(frozen=True)
class LearnedCriticResult(_JsonDTO):
    critic: str
    scale: str
    target_path: str
    candidate_id: str
    findings: tuple[str, ...] = ()
    features: Mapping[str, float] = MappingProxyType({})
    passed: bool | None = None
    score: float | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _required_text(self.critic, "critic")
        if self.scale == "mechanical":
            raise ProtocolError("Python critics may not claim the mechanical scale")
        _required_text(self.scale, "scale")
        _required_text(self.target_path, "target_path")
        _required_text(self.candidate_id, "candidate_id")
        object.__setattr__(
            self,
            "features",
            MappingProxyType(
                {
                    _required_text(key, "feature name"): float(value)
                    for key, value in sorted(self.features.items())
                }
            ),
        )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ProtocolError("confidence must be between zero and one")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LearnedCriticResult":
        try:
            return cls(
                critic=value["critic"],
                scale=value["scale"],
                target_path=value["target_path"],
                candidate_id=value["candidate_id"],
                findings=tuple(value.get("findings", ())),
                features=value.get("features", {}),
                passed=value.get("passed"),
                score=value.get("score"),
                confidence=value.get("confidence"),
            )
        except KeyError as error:
            raise ProtocolError(f"critic result lacks {error.args[0]}") from error

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "critic": self.critic,
            "scale": self.scale,
            "target_path": self.target_path,
            "candidate_id": self.candidate_id,
            "findings": list(self.findings),
            "features": dict(self.features),
        }
        if self.passed is not None:
            value["passed"] = self.passed
        if self.score is not None:
            value["score"] = self.score
        if self.confidence is not None:
            value["confidence"] = self.confidence
        return value


@dataclass(frozen=True)
class SelectionResponse(_JsonDTO):
    request_id: str
    producer: str
    selected_candidate_id: str
    reason: str
    critic_results: tuple[LearnedCriticResult, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.request_id, "request_id")
        _required_text(self.producer, "producer")
        _required_text(self.selected_candidate_id, "selected_candidate_id")
        _required_text(self.reason, "reason")

    @classmethod
    def create(
        cls,
        request: SelectionRequest,
        *,
        producer: str,
        selected_candidate_id: str,
        reason: str,
        critic_results: Sequence[LearnedCriticResult] = (),
    ) -> "SelectionResponse":
        response = cls(
            request_id=request.request_id,
            producer=producer,
            selected_candidate_id=selected_candidate_id,
            reason=reason,
            critic_results=tuple(critic_results),
        )
        response.validate_for(request)
        return response

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SelectionResponse":
        data = _message(value, "selection_response")
        try:
            return cls(
                request_id=data["request_id"],
                producer=data["producer"],
                selected_candidate_id=data["selected_candidate_id"],
                reason=data["reason"],
                critic_results=tuple(
                    LearnedCriticResult.from_dict(item)
                    for item in _array(data.get("critic_results", ()), "critic_results")
                ),
            )
        except KeyError as error:
            raise ProtocolError(f"selection response lacks {error.args[0]}") from error

    @classmethod
    def from_json(cls, value: str) -> "SelectionResponse":
        return cls.from_dict(_load_json(value))

    def validate_for(self, request: SelectionRequest) -> None:
        if self.request_id != request.request_id:
            raise ProtocolError("selection response belongs to another request")
        allowed = request.candidate_ids + (request.original_candidate_id,)
        if self.selected_candidate_id not in allowed:
            raise ProtocolError("selection names an unknown candidate")
        if any(
            result.candidate_id not in request.candidate_ids
            for result in self.critic_results
        ):
            raise ProtocolError("critic result names an unknown candidate")
        identities = [
            (result.candidate_id, result.critic, result.scale)
            for result in self.critic_results
        ]
        if len(set(identities)) != len(identities):
            raise ProtocolError("selection response has duplicate critic results")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "selection_response",
            "request_id": self.request_id,
            "producer": self.producer,
            "selected_candidate_id": self.selected_candidate_id,
            "reason": self.reason,
            "critic_results": [result.to_dict() for result in self.critic_results],
        }


def _sha256(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError("source_patch must be a non-empty string")
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"
