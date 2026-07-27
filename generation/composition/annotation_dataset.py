"""Split-safe, Partitura-bound analytical supervision for score representations."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from generation.composition.observation_dataset import (
    ObservationDataset,
    ObservationDatasetError,
    _canonical_digest,
    _file_digest,
    _identifier,
    _list,
    _mapping,
    _positive_integer,
    _safe_relative,
    _string,
)


class AnnotationDatasetError(ValueError):
    """Raised when annotation semantics, bindings, or examples are invalid."""


_AVAILABILITY = {"supported", "unavailable"}
_TARGET_KINDS = {"representation", "critic"}
_SPLITS = {"train", "validation", "test"}
_ANNOTATION_PROJECTOR = "partitura-annotation-observation-v1-r1"


@dataclass(frozen=True)
class AnnotationProfileSpec:
    collection_id: str
    profile: str
    input_kinds: tuple[str, ...]
    reference_only_kinds: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object) -> AnnotationProfileSpec:
        data = _mapping(value, "annotation profile")
        input_kinds = tuple(
            _identifier(item, "annotation input kind")
            for item in _list(data.get("input_kinds"), "annotation input kinds")
        )
        references = tuple(
            _identifier(item, "reference-only annotation kind")
            for item in _list(
                data.get("reference_only_kinds"),
                "reference-only annotation kinds",
            )
        )
        if not input_kinds or set(input_kinds) & set(references):
            raise AnnotationDatasetError(
                "profile input and reference-only kinds must be non-empty and disjoint"
            )
        return cls(
            collection_id=_identifier(
                data.get("collection_id"), "profile collection_id"
            ),
            profile=_identifier(data.get("profile"), "annotation profile name"),
            input_kinds=input_kinds,
            reference_only_kinds=references,
        )


@dataclass(frozen=True)
class AnnotationTargetSpec:
    id: str
    kind: str
    availability: str
    source: str
    meaning: str
    unit: str
    metric: str | None
    baselines: tuple[str, ...]
    reason: str | None
    representation_target: str | None

    @classmethod
    def from_dict(cls, value: object) -> AnnotationTargetSpec:
        data = _mapping(value, "annotation target")
        kind = _string(data.get("kind"), "annotation target kind")
        availability = _string(
            data.get("availability"), "annotation target availability"
        )
        if kind not in _TARGET_KINDS or availability not in _AVAILABILITY:
            raise AnnotationDatasetError("annotation target kind or availability is invalid")
        metric = data.get("metric")
        if metric is not None:
            metric = _identifier(metric, "annotation target metric")
        baselines = tuple(
            _identifier(item, "baseline name")
            for item in _list(data.get("baselines"), "target baselines")
        )
        reason = data.get("reason")
        if reason is not None:
            reason = _string(reason, "unavailable target reason")
        if availability == "supported" and (metric is None or not baselines):
            raise AnnotationDatasetError(
                "supported annotation targets require a metric and baselines"
            )
        if availability == "unavailable" and (metric is not None or baselines or not reason):
            raise AnnotationDatasetError(
                "unavailable targets require a reason and no metric or baselines"
            )
        representation_target = data.get("representation_target")
        if representation_target is not None:
            representation_target = _identifier(
                representation_target, "representation target"
            )
        return cls(
            id=_identifier(data.get("id"), "annotation target id"),
            kind=kind,
            availability=availability,
            source=_string(data.get("source"), "annotation target source"),
            meaning=_string(data.get("meaning"), "annotation target meaning"),
            unit=_string(data.get("unit"), "annotation target unit"),
            metric=metric,
            baselines=baselines,
            reason=reason,
            representation_target=representation_target,
        )


@dataclass(frozen=True)
class AnnotationDatasetSpec:
    dataset_id: str
    annotation_schema_version: int
    data_root: PurePosixPath
    observation_manifest: PurePosixPath
    expected_observation_manifest_digest: str
    output_root: PurePosixPath
    profiles: tuple[AnnotationProfileSpec, ...]
    targets: tuple[AnnotationTargetSpec, ...]
    minimum_coverage: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> AnnotationDatasetSpec:
        spec_path = Path(path)
        try:
            document = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationDatasetError(
                f"cannot read annotation dataset spec {spec_path}: {error}"
            ) from error
        data = _mapping(document, "annotation dataset spec")
        if data.get("schema_version") != 1:
            raise AnnotationDatasetError(
                "annotation dataset spec schema_version must be 1"
            )
        profiles = tuple(
            AnnotationProfileSpec.from_dict(item)
            for item in _list(data.get("profiles"), "annotation profiles")
        )
        targets = tuple(
            AnnotationTargetSpec.from_dict(item)
            for item in _list(data.get("targets"), "annotation targets")
        )
        cls._unique([profile.collection_id for profile in profiles], "profile collection")
        cls._unique([target.id for target in targets], "annotation target")
        output_root = _safe_relative(data.get("output_root"), "output_root")
        if output_root.parts[0] != "outputs":
            raise AnnotationDatasetError("output_root must stay under outputs/")
        return cls(
            dataset_id=_identifier(data.get("dataset_id"), "dataset_id"),
            annotation_schema_version=_positive_integer(
                data.get("annotation_schema_version"),
                "annotation_schema_version",
            ),
            data_root=_safe_relative(data.get("data_root"), "data_root"),
            observation_manifest=_safe_relative(
                data.get("observation_manifest"), "observation_manifest"
            ),
            expected_observation_manifest_digest=_string(
                data.get("expected_observation_manifest_digest"),
                "expected_observation_manifest_digest",
            ),
            output_root=output_root,
            profiles=profiles,
            targets=targets,
            minimum_coverage=_mapping(
                data.get("minimum_coverage"), "minimum_coverage"
            ),
            raw=data,
        )

    @staticmethod
    def _unique(values: list[str], label: str) -> None:
        if not values or len(set(values)) != len(values):
            raise AnnotationDatasetError(f"{label} ids must be non-empty and unique")

    @property
    def digest(self) -> str:
        return _canonical_digest(self.raw)

    @property
    def supported_targets(self) -> tuple[AnnotationTargetSpec, ...]:
        return tuple(
            target for target in self.targets if target.availability == "supported"
        )

    @property
    def profile_by_collection(self) -> Mapping[str, AnnotationProfileSpec]:
        return {profile.collection_id: profile for profile in self.profiles}


@dataclass(frozen=True)
class AnnotationObservation:
    data: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> AnnotationObservation:
        data = _mapping(value, "annotation observation")
        if data.get("schema_version") != 1:
            raise AnnotationDatasetError(
                "annotation observation schema_version must be 1"
            )
        if data.get("projector") != _ANNOTATION_PROJECTOR:
            raise AnnotationDatasetError(
                "annotation observation projector revision is unsupported"
            )
        claimed = _string(
            data.get("annotation_observation_digest"),
            "annotation_observation_digest",
        )
        payload = dict(data)
        payload.pop("annotation_observation_digest", None)
        actual = _canonical_digest(payload)
        if claimed != actual:
            raise AnnotationDatasetError(
                f"annotation observation digest mismatch: {claimed} != {actual}"
            )
        examples = _list(data.get("examples"), "annotation examples")
        ids: set[str] = set()
        for raw_example in examples:
            example = _mapping(raw_example, "annotation example")
            example_id = _string(example.get("example_id"), "example_id")
            if example_id in ids:
                raise AnnotationDatasetError(f"duplicate example_id: {example_id}")
            ids.add(example_id)
            cls._validate_example(example)
        return cls(data=json.loads(json.dumps(data)))

    @staticmethod
    def _validate_example(example: Mapping[str, Any]) -> None:
        _identifier(example.get("target"), "example target")
        _string(example.get("label"), "example label")
        names = _list(example.get("feature_names"), "feature_names")
        values = _list(example.get("features"), "features")
        if not names or len(names) != len(values) or len(set(names)) != len(names):
            raise AnnotationDatasetError(
                "feature_names must be non-empty, unique, and align with features"
            )
        for value in values:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AnnotationDatasetError("example features must be numeric")
            if not math.isfinite(float(value)):
                raise AnnotationDatasetError("example features must be finite")
        _mapping(example.get("scope"), "example scope")
        _mapping(example.get("provenance"), "example provenance")

    @classmethod
    def load(cls, path: str | Path) -> AnnotationObservation:
        observation_path = Path(path)
        try:
            return cls.from_dict(
                json.loads(observation_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationDatasetError(
                f"cannot read annotation observation {observation_path}: {error}"
            ) from error

    @property
    def digest(self) -> str:
        return str(self.data["annotation_observation_digest"])

    @property
    def score_observation_digest(self) -> str:
        return str(self.data["score_observation_digest"])

    @property
    def summary(self) -> Mapping[str, Any]:
        return _mapping(self.data["summary"], "annotation summary")


@dataclass(frozen=True)
class AnnotationDatasetBuilder:
    spec: AnnotationDatasetSpec
    project_root: Path
    partitura_bin: Path
    progress: Callable[[str], None] = print
    timeout_seconds: float = 300.0

    def __init__(
        self,
        spec: AnnotationDatasetSpec,
        *,
        project_root: str | Path,
        partitura_bin: str | Path,
        progress: Callable[[str], None] = print,
        timeout_seconds: float = 300.0,
    ):
        if timeout_seconds <= 0:
            raise AnnotationDatasetError("timeout_seconds must be positive")
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "project_root", Path(project_root).resolve())
        object.__setattr__(self, "partitura_bin", Path(partitura_bin).resolve())
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "timeout_seconds", timeout_seconds)

    @property
    def output_root(self) -> Path:
        return self.project_root.joinpath(*self.spec.output_root.parts)

    @property
    def data_root(self) -> Path:
        return self.project_root.joinpath(*self.spec.data_root.parts)

    def build(self, *, jobs: int = 2) -> Mapping[str, Any]:
        if jobs < 1:
            raise AnnotationDatasetError("jobs must be positive")
        dataset = self._observation_dataset()
        profile_map = self.spec.profile_by_collection
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        raw_records = _list(dataset.manifest["records"], "observation records")
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            future_targets = {
                executor.submit(
                    self._build_record,
                    _mapping(raw, "observation record"),
                    profile_map,
                    dataset.root,
                ): _mapping(raw, "observation record").get("target_id")
                for raw in raw_records
            }
            for future in as_completed(future_targets):
                target_id = str(future_targets[future])
                try:
                    records.append(future.result())
                except (
                    AnnotationDatasetError,
                    ObservationDatasetError,
                    OSError,
                ) as error:
                    failures.append({"target_id": target_id, "error": str(error)})
                    self.progress(f"[{target_id}] failed: {error}")
        manifest = self._manifest(
            dataset.manifest,
            sorted(records, key=lambda record: record["target_id"]),
            sorted(failures, key=lambda failure: failure["target_id"]),
        )
        self._write_json(self.output_root / "manifest.json", manifest)
        return manifest

    def _observation_dataset(self) -> ObservationDataset:
        path = self.project_root.joinpath(*self.spec.observation_manifest.parts)
        dataset = ObservationDataset.load(path)
        actual = dataset.manifest.get("manifest_digest")
        expected = self.spec.expected_observation_manifest_digest
        if actual != expected:
            raise AnnotationDatasetError(
                f"observation manifest digest mismatch: {actual} != {expected}"
            )
        return dataset

    def _build_record(
        self,
        record: Mapping[str, Any],
        profiles: Mapping[str, AnnotationProfileSpec],
        observation_root: Path,
    ) -> dict[str, Any]:
        target_id = _string(record.get("target_id"), "target_id")
        collection_id = _string(record.get("collection_id"), "collection_id")
        profile = profiles.get(collection_id)
        if profile is None:
            raise AnnotationDatasetError(
                f"no annotation profile for collection {collection_id}"
            )
        annotations = self._profile_annotations(record, profile)
        observation_file = _safe_relative(
            record.get("observation_file"), "observation_file"
        )
        score_observation_path = observation_root.joinpath(*observation_file.parts)
        output_path = self.output_root / "observations" / f"{target_id}.json"
        projection = self._existing(
            output_path,
            score_digest=_string(
                record.get("observation_digest"), "observation_digest"
            ),
            profile=profile.profile,
            annotations=annotations,
        )
        if projection is None:
            self.progress(f"[{target_id}] binding {profile.profile}")
            projection = self._project(
                score_observation_path,
                profile,
                annotations,
            )
            self._write_json(output_path, projection.data)
        else:
            self.progress(f"[{target_id}] verified cached annotation observation")
        warning_counts = Counter(
            _identifier(
                _mapping(raw_warning, "annotation warning").get("code"),
                "annotation warning code",
            )
            for raw_warning in _list(
                projection.data.get("warnings"), "annotation warnings"
            )
        )
        return {
            "target_id": target_id,
            "lineage_id": _string(record.get("lineage_id"), "lineage_id"),
            "split": _string(record.get("split"), "split"),
            "profile": profile.profile,
            "score_observation_digest": projection.score_observation_digest,
            "annotation_observation_file": output_path.relative_to(
                self.output_root
            ).as_posix(),
            "annotation_observation_digest": projection.digest,
            "target_counts": dict(projection.summary["target_counts"]),
            "warning_count": projection.summary["warning_count"],
            "warning_code_counts": dict(sorted(warning_counts.items())),
            "binding_failure_count": projection.summary["binding_failure_count"],
            "failed_audit_count": projection.summary["failed_audit_count"],
        }

    def _profile_annotations(
        self,
        record: Mapping[str, Any],
        profile: AnnotationProfileSpec,
    ) -> list[dict[str, str]]:
        annotations = [
            _mapping(item, "annotation provenance")
            for item in _list(record.get("annotations"), "annotations")
        ]
        selected = [
            {
                "kind": str(annotation["kind"]),
                "path": str(annotation["path"]),
                "digest": str(annotation["digest"]),
            }
            for annotation in annotations
            if annotation.get("kind") in profile.input_kinds
        ]
        present = {annotation["kind"] for annotation in selected}
        missing = set(profile.input_kinds) - present
        if missing:
            raise AnnotationDatasetError(
                f"{record.get('target_id')} lacks annotation kinds: {sorted(missing)}"
            )
        for annotation in selected:
            relative = _safe_relative(annotation["path"], "annotation path")
            actual = _file_digest(self.data_root.joinpath(*relative.parts))
            if actual != annotation["digest"]:
                raise AnnotationDatasetError(
                    f"annotation source digest mismatch: {annotation['path']}"
                )
        return selected

    def _existing(
        self,
        path: Path,
        *,
        score_digest: str,
        profile: str,
        annotations: list[dict[str, str]],
    ) -> AnnotationObservation | None:
        if not path.is_file():
            return None
        projection = AnnotationObservation.load(path)
        expected_sources = {
            (
                annotation["kind"],
                f"{self.spec.data_root}/{annotation['path']}",
                annotation["digest"],
            )
            for annotation in annotations
        }
        actual_sources = {
            (
                str(source["kind"]),
                str(source["path"]),
                str(source["digest"]),
            )
            for source in projection.data["annotation_sources"]
        }
        if (
            projection.score_observation_digest == score_digest
            and projection.data.get("projector") == _ANNOTATION_PROJECTOR
            and projection.data.get("profile") == profile
            and actual_sources == expected_sources
        ):
            return projection
        return None

    def _project(
        self,
        score_observation_path: Path,
        profile: AnnotationProfileSpec,
        annotations: list[dict[str, str]],
    ) -> AnnotationObservation:
        command = [
            "ruby",
            str(self.partitura_bin),
            "annotation-observation",
            score_observation_path.relative_to(self.project_root).as_posix(),
            "--profile",
            profile.profile,
        ]
        for annotation in annotations:
            command.extend(
                [
                    "--annotation",
                    f"{annotation['kind']}={self.spec.data_root}/{annotation['path']}",
                ]
            )
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise AnnotationDatasetError(
                f"Partitura annotation projection timed out for {score_observation_path}"
            ) from error
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise AnnotationDatasetError(
                f"Partitura annotation projection failed: {detail or 'unknown error'}"
            )
        try:
            return AnnotationObservation.from_dict(json.loads(result.stdout))
        except json.JSONDecodeError as error:
            raise AnnotationDatasetError(
                "Partitura returned invalid annotation-observation JSON"
            ) from error

    def _manifest(
        self,
        observation_manifest: Mapping[str, Any],
        records: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        coverage = self._coverage(records, failures)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": self.spec.dataset_id,
            "spec_digest": self.spec.digest,
            "annotation_schema_version": self.spec.annotation_schema_version,
            "observation_manifest_digest": observation_manifest["manifest_digest"],
            "projector": _ANNOTATION_PROJECTOR,
            "targets": [target.__dict__ for target in self.spec.targets],
            "records": records,
            "failures": failures,
            "coverage": coverage,
            "ready": all(check["passed"] for check in coverage["checks"]),
        }
        payload["manifest_digest"] = _canonical_digest(payload)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        return payload

    def _coverage(
        self,
        records: list[dict[str, Any]],
        failures: list[dict[str, Any]],
    ) -> dict[str, Any]:
        profile_counts = Counter(record["profile"] for record in records)
        target_counts: Counter[str] = Counter()
        warning_code_counts: Counter[str] = Counter()
        split_target_counts: dict[str, Counter[str]] = {
            split: Counter() for split in sorted(_SPLITS)
        }
        for record in records:
            target_counts.update(record["target_counts"])
            warning_code_counts.update(record["warning_code_counts"])
            split_target_counts[record["split"]].update(record["target_counts"])
        supported_with_examples = sum(
            target_counts[target.id] > 0 for target in self.spec.supported_targets
        )
        unavailable = sum(
            target.availability == "unavailable" for target in self.spec.targets
        )
        actual = {
            "score_count": len(records),
            "failed_score_count": len(failures),
            "binding_failure_count": sum(
                record["binding_failure_count"] for record in records
            ),
            "failed_audit_count": sum(
                record["failed_audit_count"] for record in records
            ),
            "warning_count": sum(record["warning_count"] for record in records),
            "warning_code_counts": dict(sorted(warning_code_counts.items())),
            "profile_score_counts": dict(sorted(profile_counts.items())),
            "target_example_counts": dict(sorted(target_counts.items())),
            "split_target_example_counts": {
                split: dict(sorted(counts.items()))
                for split, counts in split_target_counts.items()
            },
            "supported_targets_with_examples": supported_with_examples,
            "unavailable_target_count": unavailable,
        }
        return {**actual, "checks": self._coverage_checks(actual)}

    def _coverage_checks(
        self, actual: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        checks: list[dict[str, Any]] = []
        for key in (
            "score_count",
            "failed_score_count",
            "binding_failure_count",
            "failed_audit_count",
            "supported_targets_with_examples",
            "unavailable_target_count",
        ):
            expected = _positive_integer(
                self.spec.minimum_coverage.get(key),
                f"minimum_coverage.{key}",
                allow_zero=key in {
                    "failed_score_count",
                    "binding_failure_count",
                    "failed_audit_count",
                },
            )
            checks.append(
                {
                    "name": key,
                    "expected": expected,
                    "actual": actual[key],
                    "passed": actual[key] == expected,
                }
            )
        expected_profiles = dict(
            _mapping(
                self.spec.minimum_coverage.get("profile_score_counts"),
                "minimum_coverage.profile_score_counts",
            )
        )
        checks.append(
            {
                "name": "profile_score_counts",
                "expected": expected_profiles,
                "actual": actual["profile_score_counts"],
                "passed": actual["profile_score_counts"] == expected_profiles,
            }
        )
        return checks

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)


@dataclass(frozen=True)
class TrainingExample:
    id: str
    target: str
    label: str
    feature_names: tuple[str, ...]
    features: tuple[float, ...]
    target_id: str
    lineage_id: str
    split: str
    scope: Mapping[str, Any]
    provenance: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class AnnotationDataset:
    root: Path
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> AnnotationDataset:
        manifest_path = Path(path).resolve()
        try:
            document = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AnnotationDatasetError(
                f"cannot read annotation manifest {manifest_path}: {error}"
            ) from error
        manifest = _mapping(document, "annotation manifest")
        if manifest.get("schema_version") != 1:
            raise AnnotationDatasetError(
                "annotation manifest schema_version must be 1"
            )
        claimed = _string(manifest.get("manifest_digest"), "manifest_digest")
        payload = dict(manifest)
        payload.pop("manifest_digest", None)
        payload.pop("generated_at", None)
        actual = _canonical_digest(payload)
        if claimed != actual:
            raise AnnotationDatasetError(
                f"annotation manifest digest mismatch: {claimed} != {actual}"
            )
        return cls(root=manifest_path.parent, manifest=dict(manifest))

    def examples(
        self,
        *,
        target: str | None = None,
        split: str | None = None,
    ) -> tuple[TrainingExample, ...]:
        if split is not None and split not in _SPLITS:
            raise AnnotationDatasetError(f"unknown annotation split: {split}")
        examples: list[TrainingExample] = []
        schemas: dict[str, tuple[str, ...]] = {}
        for raw_record in _list(self.manifest.get("records"), "annotation records"):
            record = _mapping(raw_record, "annotation record")
            if split is not None and record.get("split") != split:
                continue
            projection_path = _safe_relative(
                record.get("annotation_observation_file"),
                "annotation_observation_file",
            )
            projection = AnnotationObservation.load(
                self.root.joinpath(*projection_path.parts)
            )
            if projection.digest != record.get("annotation_observation_digest"):
                raise AnnotationDatasetError(
                    f"annotation digest disagrees for {record.get('target_id')}"
                )
            examples.extend(
                self._examples_from_projection(
                    projection,
                    record,
                    target=target,
                    schemas=schemas,
                )
            )
        return tuple(sorted(examples, key=lambda example: example.id))

    @staticmethod
    def _examples_from_projection(
        projection: AnnotationObservation,
        record: Mapping[str, Any],
        *,
        target: str | None,
        schemas: dict[str, tuple[str, ...]],
    ) -> list[TrainingExample]:
        examples = []
        for raw_example in _list(projection.data["examples"], "examples"):
            example = _mapping(raw_example, "example")
            example_target = str(example["target"])
            if target is not None and example_target != target:
                continue
            names = tuple(str(name) for name in example["feature_names"])
            previous = schemas.setdefault(example_target, names)
            if previous != names:
                raise AnnotationDatasetError(
                    f"feature schema varies within target {example_target}"
                )
            examples.append(
                TrainingExample(
                    id=str(example["example_id"]),
                    target=example_target,
                    label=str(example["label"]),
                    feature_names=names,
                    features=tuple(float(value) for value in example["features"]),
                    target_id=str(record["target_id"]),
                    lineage_id=str(record["lineage_id"]),
                    split=str(record["split"]),
                    scope=dict(_mapping(example["scope"], "scope")),
                    provenance=dict(
                        _mapping(example["provenance"], "provenance")
                    ),
                    metadata=dict(_mapping(example["metadata"], "metadata")),
                )
            )
        return examples
