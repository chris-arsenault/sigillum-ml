"""Append-only run storage and exact trajectory effort accounting."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from generation.composition.benchmark import (
    BenchmarkError,
    _mapping,
)
from generation.composition.evidence import TrajectoryRecord
from generation.composition.evaluation_run import EvaluationRun, RunEffort


class EvaluationRunStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()

    def load(self) -> tuple[EvaluationRun, ...]:
        if not self.path.exists():
            return ()
        records = []
        with self.path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(
                        EvaluationRun.from_dict(
                            _mapping(json.loads(line), "evaluation run")
                        )
                    )
                except (json.JSONDecodeError, BenchmarkError) as error:
                    raise BenchmarkError(
                        f"invalid run at line {line_number}: {error}"
                    ) from error
        return tuple(records)

    def append(self, run: EvaluationRun) -> EvaluationRun:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as target:
            fcntl.flock(target.fileno(), fcntl.LOCK_EX)
            target.seek(0)
            self._validate_existing(target, run)
            target.seek(0, os.SEEK_END)
            target.write(
                json.dumps(
                    run.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
            fcntl.flock(target.fileno(), fcntl.LOCK_UN)
        return run

    @staticmethod
    def _validate_existing(target: object, run: EvaluationRun) -> None:
        for line in target:
            if not line.strip():
                continue
            existing = EvaluationRun.from_dict(
                _mapping(json.loads(line), "evaluation run")
            )
            if existing.run_id == run.run_id:
                raise BenchmarkError(f"run {run.run_id} is already stored")
            if existing.cell.key() == run.cell.key():
                raise BenchmarkError(
                    f"benchmark cell is already stored: {run.cell.key()}"
                )


def trajectory_effort(
    path: str | Path, *, model_call_count: int, wall_seconds: float
) -> RunEffort:
    records = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(
                    TrajectoryRecord.from_dict(_mapping(json.loads(line), "trajectory"))
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise BenchmarkError(
                    f"invalid trajectory at line {line_number}: {error}"
                ) from error
    candidates = sum(len(record.candidates) for record in records)
    mechanically_valid = sum(
        _mechanically_valid(assessment)
        for record in records
        for assessment in record.candidates
    )
    accepted = sum(record.decision == "accept" for record in records)
    return RunEffort.from_dict(
        {
            "candidate_count": candidates,
            "mechanically_valid_candidate_count": mechanically_valid,
            "accepted_edit_count": accepted,
            "model_call_count": model_call_count,
            "wall_seconds": wall_seconds,
        }
    )


def _mechanically_valid(assessment: object) -> int:
    evidence = _mapping(assessment, "candidate assessment")
    results = evidence.get("critic_results")
    if not isinstance(results, (list, tuple)):
        raise BenchmarkError("candidate assessment lacks critic_results")
    mechanical = [
        _mapping(result, "critic result")
        for result in results
        if _mapping(result, "critic result").get("scale") == "mechanical"
    ]
    return int(
        bool(mechanical) and all(item.get("passed") is True for item in mechanical)
    )
