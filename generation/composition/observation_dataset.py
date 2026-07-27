"""Auditable external-score projection manifests for representation learning."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class ObservationDatasetError(ValueError):
    """Raised when a projection spec, observation, or manifest is invalid."""


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SPLITS = {"train", "validation", "test"}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ObservationDatasetError(f"{label} must be an object")
    return value


def _list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ObservationDatasetError(f"{label} must be a list")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObservationDatasetError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    identifier = _string(value, label)
    if not _IDENTIFIER.fullmatch(identifier):
        raise ObservationDatasetError(f"{label} is invalid: {identifier}")
    return identifier


def _positive_integer(value: object, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ObservationDatasetError(f"{label} must be a {qualifier} integer")
    return value


def _safe_relative(value: object, label: str) -> PurePosixPath:
    path = PurePosixPath(_string(value, label))
    if (
        path.is_absolute()
        or not path.parts
        or str(path) == "."
        or ".." in path.parts
    ):
        raise ObservationDatasetError(f"{label} must be a safe relative path")
    return path


def _safe_glob(value: object, label: str) -> str:
    pattern = _string(value, label)
    path = PurePosixPath(pattern)
    if path.is_absolute() or ".." in path.parts:
        raise ObservationDatasetError(f"{label} must stay under its collection root")
    return pattern


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
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class AnnotationRule:
    kind: str
    patterns: tuple[str, ...]

    @classmethod
    def from_dict(
        cls, value: object, *, collection_id: str
    ) -> AnnotationRule:
        data = _mapping(value, f"annotation rule in {collection_id}")
        kind = _identifier(data.get("kind"), f"annotation kind in {collection_id}")
        patterns = tuple(
            _safe_glob(item, f"annotation pattern for {collection_id}:{kind}")
            for item in _list(
                data.get("patterns"), f"annotation patterns for {collection_id}:{kind}"
            )
        )
        if not patterns:
            raise ObservationDatasetError(
                f"annotation rule {collection_id}:{kind} has no patterns"
            )
        return cls(kind=kind, patterns=patterns)


@dataclass(frozen=True)
class LineageRule:
    id: str
    split: str
    score_globs: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: object, *, collection_id: str) -> LineageRule:
        data = _mapping(value, f"lineage in {collection_id}")
        lineage_id = _identifier(data.get("id"), f"lineage id in {collection_id}")
        split = _string(data.get("split"), f"split for {lineage_id}")
        if split not in _SPLITS:
            raise ObservationDatasetError(
                f"split for {lineage_id} must be train, validation, or test"
            )
        globs = tuple(
            _safe_glob(item, f"score glob for lineage {lineage_id}")
            for item in _list(
                data.get("score_globs"), f"score globs for lineage {lineage_id}"
            )
        )
        if not globs:
            raise ObservationDatasetError(f"lineage {lineage_id} has no score globs")
        return cls(id=lineage_id, split=split, score_globs=globs)

    def matches(self, path: str) -> bool:
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in self.score_globs)


@dataclass(frozen=True)
class CollectionSpec:
    id: str
    source_id: str
    source_version: Mapping[str, Any]
    root: PurePosixPath
    score_glob: str
    exclude_globs: tuple[str, ...]
    expected_score_count: int
    annotation_rules: tuple[AnnotationRule, ...]
    lineages: tuple[LineageRule, ...]

    @classmethod
    def from_dict(cls, value: object) -> CollectionSpec:
        data = _mapping(value, "collection")
        collection_id = _identifier(data.get("id"), "collection id")
        lineages = tuple(
            LineageRule.from_dict(item, collection_id=collection_id)
            for item in _list(data.get("lineages"), f"lineages in {collection_id}")
        )
        lineage_ids = [lineage.id for lineage in lineages]
        if not lineages or len(set(lineage_ids)) != len(lineage_ids):
            raise ObservationDatasetError(
                f"lineage ids in {collection_id} must be non-empty and unique"
            )
        annotations = tuple(
            AnnotationRule.from_dict(item, collection_id=collection_id)
            for item in _list(
                data.get("annotation_rules"), f"annotation rules in {collection_id}"
            )
        )
        annotation_kinds = [item.kind for item in annotations]
        if len(set(annotation_kinds)) != len(annotation_kinds):
            raise ObservationDatasetError(
                f"annotation kinds in {collection_id} must be unique"
            )
        return cls(
            id=collection_id,
            source_id=_identifier(
                data.get("source_id"), f"source id for {collection_id}"
            ),
            source_version=_mapping(
                data.get("source_version"), f"source version for {collection_id}"
            ),
            root=_safe_relative(data.get("root"), f"root for {collection_id}"),
            score_glob=_safe_glob(
                data.get("score_glob"), f"score glob for {collection_id}"
            ),
            exclude_globs=tuple(
                _safe_glob(item, f"exclude glob for {collection_id}")
                for item in _list(
                    data.get("exclude_globs"), f"exclude globs for {collection_id}"
                )
            ),
            expected_score_count=_positive_integer(
                data.get("expected_score_count"),
                f"expected score count for {collection_id}",
            ),
            annotation_rules=annotations,
            lineages=lineages,
        )


@dataclass(frozen=True)
class ProjectionTarget:
    id: str
    collection: CollectionSpec
    lineage: LineageRule
    path: Path
    relative_path: PurePosixPath

    @property
    def split(self) -> str:
        return self.lineage.split


@dataclass(frozen=True)
class ObservationDatasetSpec:
    dataset_id: str
    observation_schema_version: int
    data_root: PurePosixPath
    output_root: PurePosixPath
    training_targets: Mapping[str, Any]
    split_policy: Mapping[str, Any]
    collections: tuple[CollectionSpec, ...]
    minimum_coverage: Mapping[str, Any]
    raw: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> ObservationDatasetSpec:
        spec_path = Path(path)
        try:
            document = json.loads(spec_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ObservationDatasetError(
                f"cannot read observation dataset spec {spec_path}: {error}"
            ) from error
        data = _mapping(document, "observation dataset spec")
        if data.get("schema_version") != 1:
            raise ObservationDatasetError(
                "observation dataset spec schema_version must be 1"
            )
        collections = tuple(
            CollectionSpec.from_dict(item)
            for item in _list(data.get("collections"), "collections")
        )
        collection_ids = [collection.id for collection in collections]
        if not collections or len(set(collection_ids)) != len(collection_ids):
            raise ObservationDatasetError(
                "collection ids must be non-empty and unique"
            )
        output_root = _safe_relative(data.get("output_root"), "output_root")
        if output_root.parts[0] != "outputs":
            raise ObservationDatasetError("output_root must stay under outputs/")
        return cls(
            dataset_id=_identifier(data.get("dataset_id"), "dataset_id"),
            observation_schema_version=_positive_integer(
                data.get("observation_schema_version"),
                "observation_schema_version",
            ),
            data_root=_safe_relative(data.get("data_root"), "data_root"),
            output_root=output_root,
            training_targets=_mapping(
                data.get("training_targets"), "training_targets"
            ),
            split_policy=_mapping(data.get("split_policy"), "split_policy"),
            collections=collections,
            minimum_coverage=_mapping(
                data.get("minimum_coverage"), "minimum_coverage"
            ),
            raw=data,
        )

    @property
    def digest(self) -> str:
        return _canonical_digest(self.raw)

    def discover(self, project_root: str | Path) -> tuple[ProjectionTarget, ...]:
        root = Path(project_root).resolve()
        data_root = root.joinpath(*self.data_root.parts)
        targets: list[ProjectionTarget] = []
        seen_paths: set[Path] = set()
        for collection in self.collections:
            targets.extend(self._discover_collection(data_root, collection))
        for target in targets:
            if target.path in seen_paths:
                raise ObservationDatasetError(
                    f"score is selected more than once: {target.path}"
                )
            seen_paths.add(target.path)
        return tuple(sorted(targets, key=lambda target: target.id))

    def _discover_collection(
        self, data_root: Path, collection: CollectionSpec
    ) -> list[ProjectionTarget]:
        root = data_root.joinpath(*collection.root.parts)
        if not root.is_dir():
            raise ObservationDatasetError(
                f"collection root is unavailable: {root}"
            )
        paths = [
            path
            for path in root.glob(collection.score_glob)
            if path.is_file()
            and not self._excluded(path.relative_to(root), collection.exclude_globs)
        ]
        if len(paths) != collection.expected_score_count:
            raise ObservationDatasetError(
                f"{collection.id} expected {collection.expected_score_count} scores, "
                f"found {len(paths)}"
            )
        return [
            self._target(root, collection, path)
            for path in sorted(paths)
        ]

    @staticmethod
    def _excluded(path: Path, patterns: Iterable[str]) -> bool:
        value = path.as_posix()
        return any(fnmatch.fnmatchcase(value, pattern) for pattern in patterns)

    @staticmethod
    def _target(
        root: Path, collection: CollectionSpec, path: Path
    ) -> ProjectionTarget:
        relative = PurePosixPath(path.relative_to(root).as_posix())
        lineages = [
            lineage
            for lineage in collection.lineages
            if lineage.matches(str(relative))
        ]
        if len(lineages) != 1:
            raise ObservationDatasetError(
                f"{collection.id}:{relative} must match exactly one lineage, "
                f"matched {[lineage.id for lineage in lineages]}"
            )
        identity = hashlib.sha256(
            f"{collection.source_id}\0{relative}".encode()
        ).hexdigest()[:24]
        return ProjectionTarget(
            id=f"score_{identity}",
            collection=collection,
            lineage=lineages[0],
            path=path,
            relative_path=relative,
        )


@dataclass(frozen=True)
class ScoreObservation:
    data: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: object) -> ScoreObservation:
        data = dict(_mapping(value, "score observation"))
        if data.get("schema_version") != 1:
            raise ObservationDatasetError(
                "score observation schema_version must be 1"
            )
        claimed = _string(data.get("observation_digest"), "observation_digest")
        if not _DIGEST.fullmatch(claimed):
            raise ObservationDatasetError("observation_digest must be SHA-256")
        digest_payload = dict(data)
        digest_payload.pop("observation_digest")
        actual = _canonical_digest(digest_payload)
        if claimed != actual:
            raise ObservationDatasetError(
                f"score observation digest mismatch: {claimed} != {actual}"
            )
        source = _mapping(data.get("source"), "score observation source")
        for key in ("source_digest", "document_digest"):
            digest = _string(source.get(key), f"score observation {key}")
            if not _DIGEST.fullmatch(digest):
                raise ObservationDatasetError(
                    f"score observation {key} must be SHA-256"
                )
        score = _mapping(data.get("score"), "score observation score")
        for key in ("parts", "measures", "timed_events"):
            _list(score.get(key), f"score observation score.{key}")
        summary = _mapping(data.get("summary"), "score observation summary")
        for key in (
            "part_count",
            "measure_count",
            "event_count",
            "pitched_note_count",
            "warning_count",
        ):
            _positive_integer(
                summary.get(key),
                f"score observation summary.{key}",
                allow_zero=True,
            )
        _list(data.get("warnings"), "score observation warnings")
        return cls(data=json.loads(json.dumps(data)))

    @classmethod
    def load(cls, path: str | Path) -> ScoreObservation:
        observation_path = Path(path)
        try:
            return cls.from_dict(
                json.loads(observation_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ObservationDatasetError(
                f"cannot read score observation {observation_path}: {error}"
            ) from error

    @property
    def digest(self) -> str:
        return str(self.data["observation_digest"])

    @property
    def source_digest(self) -> str:
        return str(_mapping(self.data["source"], "source")["source_digest"])

    @property
    def summary(self) -> Mapping[str, Any]:
        return _mapping(self.data["summary"], "summary")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.data))


class ObservationDatasetBuilder:
    def __init__(
        self,
        spec: ObservationDatasetSpec,
        *,
        project_root: str | Path,
        partitura_bin: str | Path,
        progress: Callable[[str], None] | None = None,
        timeout_seconds: float = 300.0,
    ) -> None:
        self.spec = spec
        self.project_root = Path(project_root).resolve()
        self.data_root = self.project_root.joinpath(*spec.data_root.parts)
        self.output_root = self.project_root.joinpath(*spec.output_root.parts)
        self.partitura_bin = Path(partitura_bin).resolve()
        self.progress = progress or (lambda _message: None)
        self.timeout_seconds = timeout_seconds
        if not self.partitura_bin.is_file():
            raise ObservationDatasetError(
                f"Partitura executable does not exist: {self.partitura_bin}"
            )

    def build(self, *, jobs: int = 2) -> Mapping[str, Any]:
        if jobs < 1:
            raise ObservationDatasetError("jobs must be positive")
        targets = self.spec.discover(self.project_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            pending = {
                executor.submit(self._build_target, target): target
                for target in targets
            }
            for future in as_completed(pending):
                target = pending[future]
                try:
                    records.append(future.result())
                except (
                    ObservationDatasetError,
                    OSError,
                    subprocess.SubprocessError,
                ) as error:
                    failures.append(
                        {
                            "target_id": target.id,
                            "source_id": target.collection.source_id,
                            "lineage_id": target.lineage.id,
                            "split": target.split,
                            "score_path": self._data_relative(target.path),
                            "error": str(error),
                        }
                    )
                    self.progress(f"[{target.id}] blocked: {error}")
        manifest = self._manifest(
            sorted(records, key=lambda record: record["target_id"]),
            sorted(failures, key=lambda failure: failure["target_id"]),
        )
        self._write_json(self.output_root / "manifest.json", manifest)
        return manifest

    def _build_target(self, target: ProjectionTarget) -> dict[str, Any]:
        source_digest = _file_digest(target.path)
        observation_path = self.output_root / "observations" / f"{target.id}.json"
        observation = self._existing_observation(
            observation_path, source_digest=source_digest
        )
        if observation is None:
            self.progress(f"[{target.id}] projecting {target.relative_path}")
            observation = self._project(target.path)
            if observation.source_digest != source_digest:
                raise ObservationDatasetError(
                    f"Partitura source digest disagrees for {target.path}"
                )
            self._write_json(observation_path, observation.to_dict())
        else:
            self.progress(f"[{target.id}] verified cached observation")
        annotations = self._annotations(target)
        return {
            "target_id": target.id,
            "collection_id": target.collection.id,
            "source_id": target.collection.source_id,
            "source_version": dict(target.collection.source_version),
            "lineage_id": target.lineage.id,
            "split": target.split,
            "score_path": self._data_relative(target.path),
            "source_digest": source_digest,
            "observation_file": observation_path.relative_to(
                self.output_root
            ).as_posix(),
            "observation_digest": observation.digest,
            "summary": dict(observation.summary),
            "annotations": annotations,
        }

    def _existing_observation(
        self, path: Path, *, source_digest: str
    ) -> ScoreObservation | None:
        if not path.is_file():
            return None
        observation = ScoreObservation.load(path)
        if (
            observation.source_digest == source_digest
            and observation.data["schema_version"]
            == self.spec.observation_schema_version
        ):
            return observation
        return None

    def _project(self, path: Path) -> ScoreObservation:
        try:
            command = subprocess.run(
                ("ruby", str(self.partitura_bin), "score-observation", str(path)),
                cwd=self.project_root,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise ObservationDatasetError(
                f"Partitura timed out after {self.timeout_seconds:g}s: {path}"
            ) from error
        if command.returncode != 0:
            detail = command.stderr.strip() or command.stdout.strip()
            raise ObservationDatasetError(
                f"Partitura rejected {path}: {detail or 'unknown error'}"
            )
        try:
            return ScoreObservation.from_dict(json.loads(command.stdout))
        except json.JSONDecodeError as error:
            raise ObservationDatasetError(
                f"Partitura returned invalid JSON for {path}"
            ) from error

    def _annotations(self, target: ProjectionTarget) -> list[dict[str, Any]]:
        collection_root = self.data_root.joinpath(*target.collection.root.parts)
        context = {
            "directory": target.relative_path.parent.as_posix(),
            "stem": target.relative_path.stem,
        }
        records: list[dict[str, Any]] = []
        for rule in target.collection.annotation_rules:
            paths: set[Path] = set()
            for template in rule.patterns:
                pattern = template.format(**context)
                paths.update(
                    path for path in collection_root.glob(pattern) if path.is_file()
                )
            for path in sorted(paths):
                records.append(self._annotation_record(path, kind=rule.kind))
        return records

    def _annotation_record(self, path: Path, *, kind: str) -> dict[str, Any]:
        record: dict[str, Any] = {
            "kind": kind,
            "path": self._data_relative(path),
            "digest": _file_digest(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() in {".csv", ".txt"}:
            with path.open(encoding="utf-8-sig", errors="replace") as source:
                lines = sum(1 for line in source if line.strip())
            record["rows"] = max(0, lines - int(path.suffix.lower() == ".csv"))
        return record

    def _data_relative(self, path: Path) -> str:
        return path.relative_to(self.data_root).as_posix()

    def _manifest(
        self, records: list[dict[str, Any]], failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        coverage = self._coverage(records, failures)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "dataset_id": self.spec.dataset_id,
            "spec_digest": self.spec.digest,
            "observation_schema_version": self.spec.observation_schema_version,
            "projector": "partitura-score-observation-v1",
            "records": records,
            "failures": failures,
            "coverage": coverage,
            "ready": all(check["passed"] for check in coverage["checks"]),
        }
        payload["manifest_digest"] = _canonical_digest(payload)
        payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        return payload

    def _coverage(
        self, records: list[dict[str, Any]], failures: list[dict[str, Any]]
    ) -> dict[str, Any]:
        split_counts = Counter(record["split"] for record in records)
        source_counts = Counter(record["source_id"] for record in records)
        annotated = sum(bool(record["annotations"]) for record in records)
        actual = {
            "score_count": len(records),
            "failed_score_count": len(failures),
            "source_score_counts": dict(sorted(source_counts.items())),
            "split_score_counts": dict(sorted(split_counts.items())),
            "lineage_count": len({record["lineage_id"] for record in records}),
            "scores_with_annotations": annotated,
            "annotation_file_count": sum(
                len(record["annotations"]) for record in records
            ),
            "annotation_row_count": sum(
                annotation.get("rows", 0)
                for record in records
                for annotation in record["annotations"]
            ),
            "part_count": sum(record["summary"]["part_count"] for record in records),
            "measure_count": sum(
                record["summary"]["measure_count"] for record in records
            ),
            "event_count": sum(
                record["summary"]["event_count"] for record in records
            ),
            "pitched_note_count": sum(
                record["summary"]["pitched_note_count"] for record in records
            ),
            "warning_count": sum(
                record["summary"]["warning_count"] for record in records
            ),
        }
        checks = self._coverage_checks(actual)
        return {**actual, "checks": checks}

    def _coverage_checks(
        self, actual: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        required = self.spec.minimum_coverage
        checks: list[dict[str, Any]] = []
        for key in (
            "score_count",
            "failed_score_count",
            "lineage_count",
            "scores_with_annotations",
        ):
            expected = _positive_integer(
                required.get(key),
                f"minimum_coverage.{key}",
                allow_zero=key == "failed_score_count",
            )
            observed = actual[key]
            checks.append(
                {
                    "name": key,
                    "expected": expected,
                    "actual": observed,
                    "passed": observed == expected,
                }
            )
        for key in ("source_score_counts", "split_score_counts"):
            expected_mapping = dict(
                _mapping(required.get(key), f"minimum_coverage.{key}")
            )
            observed_mapping = actual[key]
            checks.append(
                {
                    "name": key,
                    "expected": expected_mapping,
                    "actual": observed_mapping,
                    "passed": observed_mapping == expected_mapping,
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
class ObservationDataset:
    root: Path
    manifest: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> ObservationDataset:
        manifest_path = Path(path).resolve()
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ObservationDatasetError(
                f"cannot read observation dataset manifest {manifest_path}: {error}"
            ) from error
        manifest = dict(_mapping(data, "observation dataset manifest"))
        if manifest.get("schema_version") != 1:
            raise ObservationDatasetError(
                "observation dataset manifest schema_version must be 1"
            )
        claimed = _string(manifest.get("manifest_digest"), "manifest_digest")
        if not _DIGEST.fullmatch(claimed):
            raise ObservationDatasetError("manifest_digest must be SHA-256")
        digest_payload = dict(manifest)
        digest_payload.pop("manifest_digest")
        digest_payload.pop("generated_at", None)
        actual = _canonical_digest(digest_payload)
        if claimed != actual:
            raise ObservationDatasetError(
                f"dataset manifest digest mismatch: {claimed} != {actual}"
            )
        _list(manifest.get("records"), "observation dataset records")
        _list(manifest.get("failures"), "observation dataset failures")
        return cls(root=manifest_path.parent, manifest=manifest)

    def observations(
        self, *, split: str | None = None
    ) -> tuple[ScoreObservation, ...]:
        if split is not None and split not in _SPLITS:
            raise ObservationDatasetError(f"unknown dataset split: {split}")
        observations = []
        for raw_record in _list(self.manifest["records"], "records"):
            record = _mapping(raw_record, "observation record")
            if split is not None and record.get("split") != split:
                continue
            relative = _safe_relative(
                record.get("observation_file"), "observation_file"
            )
            observation = ScoreObservation.load(
                self.root.joinpath(*relative.parts)
            )
            if observation.digest != record.get("observation_digest"):
                raise ObservationDatasetError(
                    f"observation digest disagrees for {record.get('target_id')}"
                )
            observations.append(observation)
        return tuple(observations)
