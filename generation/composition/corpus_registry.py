"""Validated source registry and integrity-preserving corpus acquisition."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse


class CorpusRegistryError(ValueError):
    """Raised when a registry or fetched artifact violates its contract."""


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_DIGEST = re.compile(r"^(md5|sha256):([0-9a-f]+)$")
_DIGEST_LENGTHS = {"md5": 32, "sha256": 64}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CorpusRegistryError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusRegistryError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    identifier = _string(value, label)
    if not _IDENTIFIER.fullmatch(identifier):
        raise CorpusRegistryError(f"{label} is not a valid identifier: {identifier}")
    return identifier


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    path = PurePosixPath(_string(value, label))
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise CorpusRegistryError(f"{label} must be a safe relative path")
    return path


@dataclass(frozen=True)
class CorpusArtifact:
    id: str
    url: str
    filename: str
    expected_bytes: int | None = None
    expected_digest: str | None = None
    extract_to: PurePosixPath | None = None
    strip_components: int = 0

    @classmethod
    def from_dict(cls, value: object, *, source_id: str) -> CorpusArtifact:
        data = _mapping(value, f"artifact in {source_id}")
        artifact_id = _identifier(data.get("id"), f"artifact id in {source_id}")
        url = _string(data.get("url"), f"url for {source_id}:{artifact_id}")
        if urlparse(url).scheme not in {"file", "http", "https"}:
            raise CorpusRegistryError(
                f"url for {source_id}:{artifact_id} must use file, http, or https"
            )
        filename = _string(
            data.get("filename"), f"filename for {source_id}:{artifact_id}"
        )
        if Path(filename).name != filename:
            raise CorpusRegistryError(
                f"filename for {source_id}:{artifact_id} must be a basename"
            )
        expected_bytes = data.get("expected_bytes")
        if expected_bytes is not None and (
            not isinstance(expected_bytes, int) or expected_bytes <= 0
        ):
            raise CorpusRegistryError(
                f"expected_bytes for {source_id}:{artifact_id} must be positive"
            )
        expected_digest = data.get("expected_digest")
        if expected_digest is not None:
            expected_digest = _string(
                expected_digest, f"expected_digest for {source_id}:{artifact_id}"
            )
            match = _DIGEST.fullmatch(expected_digest)
            if match is None or len(match.group(2)) != _DIGEST_LENGTHS[match.group(1)]:
                raise CorpusRegistryError(
                    f"invalid expected_digest for {source_id}:{artifact_id}"
                )
        extract_value = data.get("extract_to")
        extract_to = (
            _safe_relative_path(
                extract_value, f"extract_to for {source_id}:{artifact_id}"
            )
            if extract_value is not None
            else None
        )
        strip_components = data.get("strip_components", 0)
        if not isinstance(strip_components, int) or strip_components < 0:
            raise CorpusRegistryError(
                f"strip_components for {source_id}:{artifact_id} "
                "must be a non-negative integer"
            )
        return cls(
            id=artifact_id,
            url=url,
            filename=filename,
            expected_bytes=expected_bytes,
            expected_digest=expected_digest,
            extract_to=extract_to,
            strip_components=strip_components,
        )


@dataclass(frozen=True)
class CorpusSource:
    id: str
    title: str
    homepage: str
    roles: tuple[str, ...]
    rights: Mapping[str, Any]
    version: Mapping[str, Any]
    artifacts: tuple[CorpusArtifact, ...]

    @classmethod
    def from_dict(cls, value: object) -> CorpusSource:
        data = _mapping(value, "corpus source")
        source_id = _identifier(data.get("id"), "source id")
        title = _string(data.get("title"), f"title for {source_id}")
        homepage = _string(data.get("homepage"), f"homepage for {source_id}")
        if urlparse(homepage).scheme not in {"http", "https"}:
            raise CorpusRegistryError(
                f"homepage for {source_id} must use http or https"
            )
        raw_roles = data.get("roles")
        if not isinstance(raw_roles, list) or not raw_roles:
            raise CorpusRegistryError(f"roles for {source_id} must be a non-empty list")
        roles = tuple(_identifier(role, f"role for {source_id}") for role in raw_roles)
        if len(set(roles)) != len(roles):
            raise CorpusRegistryError(f"roles for {source_id} must be unique")
        rights = _mapping(data.get("rights"), f"rights for {source_id}")
        version = _mapping(data.get("version"), f"version for {source_id}")
        raw_artifacts = data.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            raise CorpusRegistryError(
                f"artifacts for {source_id} must be a non-empty list"
            )
        artifacts = tuple(
            CorpusArtifact.from_dict(item, source_id=source_id)
            for item in raw_artifacts
        )
        artifact_ids = [artifact.id for artifact in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise CorpusRegistryError(f"artifact ids for {source_id} must be unique")
        filenames = [artifact.filename for artifact in artifacts]
        if len(set(filenames)) != len(filenames):
            raise CorpusRegistryError(f"artifact filenames for {source_id} must be unique")
        extract_paths = [
            artifact.extract_to
            for artifact in artifacts
            if artifact.extract_to is not None
        ]
        if len(set(extract_paths)) != len(extract_paths):
            raise CorpusRegistryError(
                f"artifact extraction targets for {source_id} must be unique"
            )
        return cls(
            id=source_id,
            title=title,
            homepage=homepage,
            roles=roles,
            rights=rights,
            version=version,
            artifacts=artifacts,
        )


@dataclass(frozen=True)
class CorpusSelection:
    source_id: str
    selection: str


@dataclass(frozen=True)
class UnresolvedSource:
    id: str
    title: str
    homepage: str
    status: str
    note: str


@dataclass(frozen=True)
class CorpusRegistry:
    data_root: PurePosixPath
    views: Mapping[str, tuple[CorpusSelection, ...]]
    unresolved_sources: tuple[UnresolvedSource, ...]
    sources: tuple[CorpusSource, ...]

    @classmethod
    def load(cls, path: str | Path) -> CorpusRegistry:
        registry_path = Path(path)
        try:
            document = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorpusRegistryError(
                f"cannot read corpus registry {registry_path}: {error}"
            ) from error
        data = _mapping(document, "corpus registry")
        if data.get("schema_version") != 1:
            raise CorpusRegistryError("corpus registry schema_version must be 1")
        data_root = _safe_relative_path(data.get("data_root"), "data_root")

        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise CorpusRegistryError("sources must be a non-empty list")
        sources = tuple(CorpusSource.from_dict(item) for item in raw_sources)
        source_ids = [source.id for source in sources]
        if len(set(source_ids)) != len(source_ids):
            raise CorpusRegistryError("source ids must be unique")

        raw_views = _mapping(data.get("views"), "views")
        views: dict[str, tuple[CorpusSelection, ...]] = {}
        for raw_view_id, raw_selections in raw_views.items():
            view_id = _identifier(raw_view_id, "view id")
            if not isinstance(raw_selections, list) or not raw_selections:
                raise CorpusRegistryError(
                    f"view {view_id} must have a non-empty selection list"
                )
            selections = tuple(
                cls._selection(item, view_id=view_id) for item in raw_selections
            )
            selected_ids = [item.source_id for item in selections]
            unknown = set(selected_ids) - set(source_ids)
            if unknown:
                raise CorpusRegistryError(
                    f"view {view_id} references unknown sources: {sorted(unknown)}"
                )
            if len(set(selected_ids)) != len(selected_ids):
                raise CorpusRegistryError(
                    f"view {view_id} must not repeat source ids"
                )
            views[view_id] = selections

        raw_unresolved = data.get("unresolved_sources", [])
        if not isinstance(raw_unresolved, list):
            raise CorpusRegistryError("unresolved_sources must be a list")
        unresolved = tuple(cls._unresolved(item) for item in raw_unresolved)
        unresolved_ids = [item.id for item in unresolved]
        if len(set(unresolved_ids)) != len(unresolved_ids):
            raise CorpusRegistryError("unresolved source ids must be unique")
        overlap = set(unresolved_ids) & set(source_ids)
        if overlap:
            raise CorpusRegistryError(
                f"sources and unresolved_sources overlap: {sorted(overlap)}"
            )
        return cls(
            data_root=data_root,
            views=views,
            unresolved_sources=unresolved,
            sources=sources,
        )

    @staticmethod
    def _selection(value: object, *, view_id: str) -> CorpusSelection:
        data = _mapping(value, f"selection in {view_id}")
        return CorpusSelection(
            source_id=_identifier(
                data.get("source_id"), f"source_id in view {view_id}"
            ),
            selection=_identifier(
                data.get("selection"), f"selection in view {view_id}"
            ),
        )

    @staticmethod
    def _unresolved(value: object) -> UnresolvedSource:
        data = _mapping(value, "unresolved source")
        source_id = _identifier(data.get("id"), "unresolved source id")
        return UnresolvedSource(
            id=source_id,
            title=_string(data.get("title"), f"title for {source_id}"),
            homepage=_string(data.get("homepage"), f"homepage for {source_id}"),
            status=_identifier(data.get("status"), f"status for {source_id}"),
            note=_string(data.get("note"), f"note for {source_id}"),
        )

    def source(self, source_id: str) -> CorpusSource:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise CorpusRegistryError(f"unknown corpus source: {source_id}")

    def view(self, view_id: str) -> tuple[CorpusSelection, ...]:
        try:
            return self.views[view_id]
        except KeyError as error:
            raise CorpusRegistryError(f"unknown corpus view: {view_id}") from error


class CorpusFetcher:
    """Fetch registered artifacts without interpreting their musical content."""

    def __init__(
        self,
        registry: CorpusRegistry,
        *,
        project_root: str | Path,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.registry = registry
        self.project_root = Path(project_root).resolve()
        self.data_root = self.project_root.joinpath(*registry.data_root.parts)
        self.state_path = self.data_root / "fetch-state.json"
        self.progress = progress or (lambda _message: None)

    def fetch(
        self, source_ids: Iterable[str], *, extract: bool = True
    ) -> Mapping[str, bool]:
        results: dict[str, bool] = {}
        for source_id in dict.fromkeys(source_ids):
            source = self.registry.source(source_id)
            results[source.id] = self._fetch_source(source, extract=extract)
        return results

    def status(self) -> Mapping[str, Any]:
        return self._load_state()

    def _fetch_source(self, source: CorpusSource, *, extract: bool) -> bool:
        self.progress(f"[{source.id}] fetching {len(source.artifacts)} artifact(s)")
        source_state = self._source_state(source)
        successful = True
        for artifact in source.artifacts:
            try:
                artifact_state = self._fetch_artifact(
                    source, artifact, extract=extract
                )
                source_state["artifacts"][artifact.id] = artifact_state
            except (CorpusRegistryError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
                successful = False
                message = str(error)
                self.progress(f"[{source.id}:{artifact.id}] blocked: {message}")
                source_state["artifacts"][artifact.id] = {
                    "status": "blocked",
                    "url": artifact.url,
                    "error": message,
                    "updated_at": _now(),
                }
            source_state["status"] = "complete" if successful else "blocked"
            source_state["updated_at"] = _now()
            self._store_source_state(source.id, source_state)
        return successful

    def _fetch_artifact(
        self,
        source: CorpusSource,
        artifact: CorpusArtifact,
        *,
        extract: bool,
    ) -> dict[str, Any]:
        download_directory = self.data_root / "sources" / source.id / "downloads"
        download_directory.mkdir(parents=True, exist_ok=True)
        target = download_directory / artifact.filename
        if target.exists():
            self.progress(f"[{source.id}:{artifact.id}] verifying existing download")
            digests, size = self._verify(target, artifact)
        else:
            self.progress(f"[{source.id}:{artifact.id}] downloading {artifact.url}")
            digests, size = self._download(target, artifact)

        extracted_to: str | None = None
        if extract and artifact.extract_to is not None:
            extract_root = (
                self.data_root
                / "sources"
                / source.id
                / "source"
                / Path(*artifact.extract_to.parts)
            )
            self._extract(
                target,
                extract_root,
                artifact=artifact,
                sha256=digests["sha256"],
            )
            extracted_to = str(extract_root.relative_to(self.data_root))
        self.progress(
            f"[{source.id}:{artifact.id}] complete "
            f"({size:,} bytes, sha256:{digests['sha256']})"
        )
        return {
            "status": "complete",
            "url": artifact.url,
            "download": str(target.relative_to(self.data_root)),
            "bytes": size,
            "digests": digests,
            "extracted_to": extracted_to,
            "updated_at": _now(),
        }

    def _download(
        self, target: Path, artifact: CorpusArtifact
    ) -> tuple[dict[str, str], int]:
        partial = target.with_name(f".{target.name}.part")
        if partial.exists():
            raise CorpusRegistryError(
                f"incomplete download already exists and was preserved: {partial}"
            )
        request = urllib.request.Request(
            artifact.url,
            headers={"User-Agent": "sigillum-ml-corpus-fetcher/1"},
        )
        size = 0
        next_report = 64 * 1024 * 1024
        hashers = {"md5": hashlib.md5(), "sha256": hashlib.sha256()}
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with partial.open("xb") as destination:
                    while chunk := response.read(1024 * 1024):
                        destination.write(chunk)
                        size += len(chunk)
                        for hasher in hashers.values():
                            hasher.update(chunk)
                        if size >= next_report:
                            self.progress(f"downloaded {size // (1024 * 1024):,} MiB")
                            next_report += 64 * 1024 * 1024
                    destination.flush()
                    os.fsync(destination.fileno())
            digests = {name: hasher.hexdigest() for name, hasher in hashers.items()}
            self._verify_contract(artifact, size=size, digests=digests)
            os.replace(partial, target)
            return digests, size
        except Exception:
            if partial.exists():
                self.progress(f"incomplete download preserved at {partial}")
            raise

    def _verify(
        self, path: Path, artifact: CorpusArtifact
    ) -> tuple[dict[str, str], int]:
        hashers = {"md5": hashlib.md5(), "sha256": hashlib.sha256()}
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                for hasher in hashers.values():
                    hasher.update(chunk)
        digests = {name: hasher.hexdigest() for name, hasher in hashers.items()}
        self._verify_contract(artifact, size=size, digests=digests)
        return digests, size

    @staticmethod
    def _verify_contract(
        artifact: CorpusArtifact, *, size: int, digests: Mapping[str, str]
    ) -> None:
        if artifact.expected_bytes is not None and size != artifact.expected_bytes:
            raise CorpusRegistryError(
                f"size mismatch for {artifact.id}: expected "
                f"{artifact.expected_bytes}, received {size}"
            )
        if artifact.expected_digest is not None:
            algorithm, expected = artifact.expected_digest.split(":", 1)
            if digests[algorithm] != expected:
                raise CorpusRegistryError(
                    f"{algorithm} mismatch for {artifact.id}: "
                    f"expected {expected}, received {digests[algorithm]}"
                )

    def _extract(
        self,
        archive: Path,
        target: Path,
        *,
        artifact: CorpusArtifact,
        sha256: str,
    ) -> None:
        marker = target / ".sigillum-fetch.json"
        if marker.is_file():
            try:
                marker_data = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise CorpusRegistryError(
                    f"invalid extraction marker at {marker}: {error}"
                ) from error
            if marker_data.get("archive_sha256") == sha256:
                self.progress(f"extraction already verified at {target}")
                return
            raise CorpusRegistryError(
                f"extraction target contains a marker for different bytes: {target}"
            )
        if target.exists():
            raise CorpusRegistryError(
                f"unmanaged extraction target already exists and was preserved: {target}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{target.name}.extracting-", dir=target.parent)
        )
        try:
            if zipfile.is_zipfile(archive):
                self._extract_zip(
                    archive, staging, strip_components=artifact.strip_components
                )
            elif tarfile.is_tarfile(archive):
                self._extract_tar(
                    archive, staging, strip_components=artifact.strip_components
                )
            else:
                raise CorpusRegistryError(f"unsupported archive format: {archive}")
            marker_data = {
                "schema_version": 1,
                "artifact_id": artifact.id,
                "archive_sha256": sha256,
                "extracted_at": _now(),
            }
            (staging / ".sigillum-fetch.json").write_text(
                json.dumps(marker_data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(staging, target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def _extract_zip(
        self, archive: Path, target: Path, *, strip_components: int
    ) -> None:
        with zipfile.ZipFile(archive) as source:
            for info in source.infolist():
                raw_path = self._archive_path(info.filename)
                relative = self._strip_path(raw_path, strip_components)
                if relative is None:
                    continue
                mode = info.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise CorpusRegistryError(
                        f"archive contains a symbolic link: {info.filename}"
                    )
                destination = self._archive_destination(target, relative)
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(info) as item, destination.open("xb") as output:
                    shutil.copyfileobj(item, output)

    def _extract_tar(
        self, archive: Path, target: Path, *, strip_components: int
    ) -> None:
        with tarfile.open(archive, mode="r:*") as source:
            for member in source:
                raw_path = self._archive_path(member.name)
                relative = self._strip_path(raw_path, strip_components)
                if relative is None:
                    continue
                if member.isdir():
                    self._archive_destination(target, relative).mkdir(
                        parents=True, exist_ok=True
                    )
                    continue
                if not member.isfile():
                    raise CorpusRegistryError(
                        f"archive contains a link or special file: {member.name}"
                    )
                item = source.extractfile(member)
                if item is None:
                    raise CorpusRegistryError(
                        f"archive member could not be read: {member.name}"
                    )
                destination = self._archive_destination(target, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with item, destination.open("xb") as output:
                    shutil.copyfileobj(item, output)

    @staticmethod
    def _archive_path(value: str) -> PurePosixPath:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise CorpusRegistryError(f"unsafe archive path: {value}")
        return path

    @staticmethod
    def _strip_path(
        path: PurePosixPath, strip_components: int
    ) -> PurePosixPath | None:
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if len(parts) <= strip_components:
            return None
        return PurePosixPath(*parts[strip_components:])

    @staticmethod
    def _archive_destination(target: Path, relative: PurePosixPath) -> Path:
        destination = target.joinpath(*relative.parts)
        try:
            destination.resolve().relative_to(target.resolve())
        except ValueError as error:
            raise CorpusRegistryError(
                f"archive path escapes extraction target: {relative}"
            ) from error
        return destination

    def _source_state(self, source: CorpusSource) -> dict[str, Any]:
        current = self._load_state().get("sources", {}).get(source.id, {})
        artifacts = current.get("artifacts", {})
        if not isinstance(artifacts, dict):
            artifacts = {}
        return {
            "title": source.title,
            "homepage": source.homepage,
            "version": dict(source.version),
            "rights": dict(source.rights),
            "status": current.get("status", "pending"),
            "artifacts": artifacts,
            "updated_at": _now(),
        }

    def _store_source_state(
        self, source_id: str, source_state: Mapping[str, Any]
    ) -> None:
        state = self._load_state()
        sources = state.setdefault("sources", {})
        sources[source_id] = dict(source_state)
        state["updated_at"] = _now()
        self.data_root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_name(f".{self.state_path.name}.tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": 1, "sources": {}}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CorpusRegistryError(
                f"cannot read fetch state {self.state_path}: {error}"
            ) from error
        if (
            not isinstance(state, dict)
            or state.get("schema_version") != 1
            or not isinstance(state.get("sources"), dict)
        ):
            raise CorpusRegistryError(f"invalid fetch state: {self.state_path}")
        return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
