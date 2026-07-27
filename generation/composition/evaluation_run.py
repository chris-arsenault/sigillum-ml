"""Versioned completed-run and edit-effort records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from generation.composition.benchmark import (
    BenchmarkCell,
    BenchmarkError,
    BenchmarkManifest,
    _canonical_digest,
    _freeze,
    _integer,
    _mapping,
    _number,
    _text,
    _thaw,
    _validate_digest,
)

RUN_SCHEMA_VERSION = 1
MEASUREMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunEffort:
    candidate_count: int
    mechanically_valid_candidate_count: int
    accepted_edit_count: int
    model_call_count: int
    wall_seconds: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RunEffort:
        effort = cls(
            candidate_count=_integer(value.get("candidate_count"), "candidate_count"),
            mechanically_valid_candidate_count=_integer(
                value.get("mechanically_valid_candidate_count"),
                "mechanically_valid_candidate_count",
            ),
            accepted_edit_count=_integer(
                value.get("accepted_edit_count"), "accepted_edit_count"
            ),
            model_call_count=_integer(
                value.get("model_call_count"), "model_call_count"
            ),
            wall_seconds=_number(value.get("wall_seconds"), "wall_seconds"),
        )
        if effort.mechanically_valid_candidate_count > effort.candidate_count:
            raise BenchmarkError("mechanically valid candidates exceed candidates")
        if effort.accepted_edit_count > effort.mechanically_valid_candidate_count:
            raise BenchmarkError("accepted edits exceed mechanically valid candidates")
        return effort

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "mechanically_valid_candidate_count": (
                self.mechanically_valid_candidate_count
            ),
            "accepted_edit_count": self.accepted_edit_count,
            "model_call_count": self.model_call_count,
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    manifest_digest: str
    benchmark_id: str
    cell: BenchmarkCell
    measurement: Mapping[str, Any]
    effort: RunEffort

    @classmethod
    def create(
        cls,
        manifest: BenchmarkManifest,
        cell: BenchmarkCell,
        measurement: Mapping[str, Any],
        effort: RunEffort,
    ) -> EvaluationRun:
        cls.validate_measurement(measurement)
        identity = {
            "manifest_digest": manifest.manifest_digest,
            **cell.to_dict(),
            "source_digest": measurement["source_digest"],
        }
        digest = _canonical_digest(identity).split(":", 1)[1]
        return cls(
            run_id=f"evaluation-run:{digest[:20]}",
            manifest_digest=manifest.manifest_digest,
            benchmark_id=manifest.benchmark_id,
            cell=cell,
            measurement=_freeze(measurement),
            effort=effort,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationRun:
        if value.get("schema_version") != RUN_SCHEMA_VERSION:
            raise BenchmarkError("unsupported evaluation run schema")
        if value.get("kind") != "whole_score_evaluation_run":
            raise BenchmarkError("invalid evaluation run kind")
        measurement = _mapping(value.get("measurement"), "measurement")
        cls.validate_measurement(measurement)
        return cls(
            run_id=_text(value.get("run_id"), "run_id"),
            manifest_digest=_validate_digest(
                value.get("manifest_digest"), "run manifest_digest"
            ),
            benchmark_id=_text(value.get("benchmark_id"), "run benchmark_id"),
            cell=BenchmarkCell(
                case_id=_text(value.get("case_id"), "run case_id"),
                strategy_id=_text(value.get("strategy_id"), "run strategy_id"),
                ablation_id=_text(value.get("ablation_id"), "run ablation_id"),
                seed=_integer(value.get("seed"), "run seed"),
            ),
            measurement=_freeze(measurement),
            effort=RunEffort.from_dict(_mapping(value.get("effort"), "effort")),
        )

    @staticmethod
    def validate_measurement(value: Mapping[str, Any]) -> None:
        if value.get("schema_version") != MEASUREMENT_SCHEMA_VERSION:
            raise BenchmarkError("unsupported Partitura measurement schema")
        if value.get("kind") != "partitura_score_measurement":
            raise BenchmarkError("invalid Partitura measurement kind")
        _validate_digest(value.get("source_digest"), "measurement source_digest")
        mechanical = _mapping(value.get("mechanical"), "measurement mechanical")
        if not isinstance(mechanical.get("valid"), bool):
            raise BenchmarkError("measurement valid flag must be boolean")
        if mechanical["valid"]:
            _validate_digest(value.get("snapshot_digest"), "snapshot_digest")
            _mapping(value.get("diagnostics"), "measurement diagnostics")
            _mapping(value.get("fingerprints"), "measurement fingerprints")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SCHEMA_VERSION,
            "kind": "whole_score_evaluation_run",
            "run_id": self.run_id,
            "manifest_digest": self.manifest_digest,
            "benchmark_id": self.benchmark_id,
            **self.cell.to_dict(),
            "measurement": _thaw(self.measurement),
            "effort": self.effort.to_dict(),
        }
