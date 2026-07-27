"""Frozen benchmark and run records for whole-score ML evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping

BENCHMARK_SCHEMA_VERSION = 1

REQUIRED_STRATEGIES = {
    "one_shot",
    "fixed_agent_like",
    "deterministic_graph",
}
REQUIRED_ABLATIONS = {
    "no_whole_score_critic",
    "no_seam_critic",
    "no_periodic_global_review",
    "no_exact_material_carry_forward",
    "no_candidate_branching",
    "no_original_as_candidate",
    "shared_critic_context",
    "expert_features_only",
    "learned_features_only",
    "no_post_export_review",
}
REQUIRED_METRICS = {
    "human.coherence",
    "human.identity",
    "human.seams",
    "human.orchestration",
    "human.reserve",
    "human.overall",
    "mechanical.validity",
    "diagnostic.requirement_binding",
    "diagnostic.identity",
    "diagnostic.seams",
    "edit_efficiency",
    "diversity",
}
HUMAN_CRITERIA = {
    "coherence",
    "identity",
    "seams",
    "orchestration",
    "reserve",
    "overall",
}


class BenchmarkError(ValueError):
    """A benchmark artifact is invalid, stale, or internally inconsistent."""


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{label} must be a non-empty string")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise BenchmarkError(f"{label} must be an integer")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"{label} must be an integer") from error
    if integer < minimum:
        raise BenchmarkError(f"{label} must be at least {minimum}")
    return integer


def _number(value: object, label: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise BenchmarkError(f"{label} must be numeric") from error
    if not math.isfinite(number) or number < minimum:
        raise BenchmarkError(f"{label} must be finite and at least {minimum}")
    return number


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise BenchmarkError(f"{label} must be an array")
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
    raise BenchmarkError(f"unsupported benchmark value {type(value).__name__}")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_digest(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _validate_digest(value: object, label: str) -> str:
    digest = _text(value, label)
    if (
        not digest.startswith("sha256:")
        or len(digest) != 71
        or any(char not in "0123456789abcdef" for char in digest[7:])
    ):
        raise BenchmarkError(f"{label} must be a sha256 digest")
    return digest


@dataclass(frozen=True)
class FrozenInput:
    path: str
    digest: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], label: str) -> FrozenInput:
        path = _text(value.get("path"), f"{label} path")
        parsed = PurePosixPath(path)
        if parsed.is_absolute() or ".." in parsed.parts:
            raise BenchmarkError(f"{label} path must be repository-relative")
        return cls(path=path, digest=_validate_digest(value.get("digest"), label))

    def verify(self, root: Path) -> None:
        resolved = root.joinpath(*PurePosixPath(self.path).parts)
        if not resolved.is_file():
            raise BenchmarkError(f"frozen input is missing: {self.path}")
        actual = _file_digest(resolved)
        if actual != self.digest:
            raise BenchmarkError(
                f"frozen input changed: {self.path} expected {self.digest}, got {actual}"
            )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "digest": self.digest}


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    brief: FrozenInput
    starting_score: FrozenInput

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BenchmarkCase:
        return cls(
            case_id=_text(value.get("case_id"), "case_id"),
            brief=FrozenInput.from_dict(_mapping(value.get("brief"), "brief"), "brief"),
            starting_score=FrozenInput.from_dict(
                _mapping(value.get("starting_score"), "starting_score"),
                "starting_score",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "brief": self.brief.to_dict(),
            "starting_score": self.starting_score.to_dict(),
        }


@dataclass(frozen=True)
class StrategySpec:
    strategy_id: str
    description: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> StrategySpec:
        return cls(
            strategy_id=_text(value.get("strategy_id"), "strategy_id"),
            description=_text(value.get("description"), "strategy description"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "strategy_id": self.strategy_id,
            "description": self.description,
        }


@dataclass(frozen=True)
class AblationSpec:
    ablation_id: str
    strategy_ids: tuple[str, ...]
    description: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AblationSpec:
        strategies = tuple(
            _text(item, "ablation strategy_id")
            for item in _array(value.get("strategy_ids"), "ablation strategy_ids")
        )
        if not strategies or len(strategies) != len(set(strategies)):
            raise BenchmarkError("ablation strategy_ids must be non-empty and unique")
        return cls(
            ablation_id=_text(value.get("ablation_id"), "ablation_id"),
            strategy_ids=strategies,
            description=_text(value.get("description"), "ablation description"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_id": self.ablation_id,
            "strategy_ids": list(self.strategy_ids),
            "description": self.description,
        }


@dataclass(frozen=True)
class BenchmarkCell:
    case_id: str
    strategy_id: str
    ablation_id: str
    seed: int

    def key(self) -> tuple[str, str, str, int]:
        return (self.case_id, self.strategy_id, self.ablation_id, self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "strategy_id": self.strategy_id,
            "ablation_id": self.ablation_id,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class BenchmarkManifest:
    benchmark_id: str
    revision: int
    title: str
    manifest_digest: str
    seeds: tuple[int, ...]
    cases: tuple[BenchmarkCase, ...]
    strategies: tuple[StrategySpec, ...]
    ablations: tuple[AblationSpec, ...]
    metrics: tuple[str, ...]
    human_criteria: tuple[str, ...]

    @classmethod
    def load(
        cls, path: str | Path, *, root: str | Path | None = None
    ) -> BenchmarkManifest:
        manifest_path = Path(path).resolve()
        try:
            raw = _mapping(
                json.loads(manifest_path.read_text(encoding="utf-8")),
                "benchmark manifest",
            )
        except json.JSONDecodeError as error:
            raise BenchmarkError(f"invalid benchmark JSON: {error}") from error
        manifest = cls.from_dict(raw)
        manifest.verify_files(
            Path(root).resolve() if root else manifest_path.parents[3]
        )
        return manifest

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BenchmarkManifest:
        if value.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
            raise BenchmarkError("unsupported benchmark schema")
        if value.get("kind") != "whole_score_benchmark":
            raise BenchmarkError("manifest kind must be whole_score_benchmark")
        expected_digest = _validate_digest(
            value.get("manifest_digest"), "manifest_digest"
        )
        payload = dict(value)
        payload.pop("manifest_digest", None)
        if _canonical_digest(payload) != expected_digest:
            raise BenchmarkError("manifest_digest does not match manifest content")
        manifest = cls(
            benchmark_id=_text(value.get("benchmark_id"), "benchmark_id"),
            revision=_integer(value.get("revision"), "revision", minimum=1),
            title=_text(value.get("title"), "benchmark title"),
            manifest_digest=expected_digest,
            seeds=tuple(
                _integer(seed, "benchmark seed")
                for seed in _array(value.get("seeds"), "seeds")
            ),
            cases=tuple(
                BenchmarkCase.from_dict(_mapping(item, "benchmark case"))
                for item in _array(value.get("cases"), "cases")
            ),
            strategies=tuple(
                StrategySpec.from_dict(_mapping(item, "strategy"))
                for item in _array(value.get("strategies"), "strategies")
            ),
            ablations=tuple(
                AblationSpec.from_dict(_mapping(item, "ablation"))
                for item in _array(value.get("ablations"), "ablations")
            ),
            metrics=tuple(
                _text(item, "metric")
                for item in _array(value.get("metrics"), "metrics")
            ),
            human_criteria=tuple(
                _text(item, "human criterion")
                for item in _array(value.get("human_criteria"), "human_criteria")
            ),
        )
        manifest.validate_contract()
        return manifest

    def validate_contract(self) -> None:
        self._unique(self.seeds, "seeds")
        self._unique((item.case_id for item in self.cases), "case ids")
        strategy_ids = tuple(item.strategy_id for item in self.strategies)
        self._unique(strategy_ids, "strategy ids")
        self._unique((item.ablation_id for item in self.ablations), "ablation ids")
        self._unique(self.metrics, "metrics")
        self._unique(self.human_criteria, "human criteria")
        if not self.seeds or not self.cases:
            raise BenchmarkError("benchmark needs at least one seed and case")
        self._require_set(REQUIRED_STRATEGIES, set(strategy_ids), "strategies")
        self._require_set(
            REQUIRED_ABLATIONS,
            {item.ablation_id for item in self.ablations},
            "ablations",
        )
        self._require_set(REQUIRED_METRICS, set(self.metrics), "metrics")
        self._require_set(HUMAN_CRITERIA, set(self.human_criteria), "human criteria")
        unknown = {
            strategy
            for ablation in self.ablations
            for strategy in ablation.strategy_ids
            if strategy not in strategy_ids
        }
        if unknown:
            raise BenchmarkError(
                f"ablations name unknown strategies: {sorted(unknown)}"
            )

    def verify_files(self, root: Path) -> None:
        for case in self.cases:
            case.brief.verify(root)
            case.starting_score.verify(root)

    def expected_cells(self) -> tuple[BenchmarkCell, ...]:
        cells = [
            BenchmarkCell(case.case_id, strategy.strategy_id, "control", seed)
            for case in self.cases
            for strategy in self.strategies
            for seed in self.seeds
        ]
        cells.extend(
            BenchmarkCell(case.case_id, strategy, ablation.ablation_id, seed)
            for case in self.cases
            for ablation in self.ablations
            for strategy in ablation.strategy_ids
            for seed in self.seeds
        )
        return tuple(cells)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "kind": "whole_score_benchmark",
            "benchmark_id": self.benchmark_id,
            "revision": self.revision,
            "title": self.title,
            "manifest_digest": self.manifest_digest,
            "seeds": list(self.seeds),
            "cases": [item.to_dict() for item in self.cases],
            "strategies": [item.to_dict() for item in self.strategies],
            "ablations": [item.to_dict() for item in self.ablations],
            "metrics": list(self.metrics),
            "human_criteria": list(self.human_criteria),
        }

    @staticmethod
    def _unique(values: Any, label: str) -> None:
        items = tuple(values)
        if len(items) != len(set(items)):
            raise BenchmarkError(f"{label} must be unique")

    @staticmethod
    def _require_set(required: set[str], actual: set[str], label: str) -> None:
        missing = required - actual
        if missing:
            raise BenchmarkError(f"benchmark lacks required {label}: {sorted(missing)}")
