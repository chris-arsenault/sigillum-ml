import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from generation.composition.corpus_registry import (
    CorpusFetcher,
    CorpusRegistry,
    CorpusRegistryError,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "corpora" / "whole_score" / "sources.json"


class CorpusRegistryTests(unittest.TestCase):
    def test_project_registry_declares_research_and_clean_views(self):
        registry = CorpusRegistry.load(REGISTRY)

        self.assertGreaterEqual(len(registry.sources), 15)
        self.assertIn("pod", {item.source_id for item in registry.view("research_full")})
        self.assertNotIn(
            "pod",
            {item.source_id for item in registry.view("redistributable_clean")},
        )
        self.assertEqual(
            {item.id for item in registry.unresolved_sources},
            {"kernscores", "musedata_full"},
        )

    def test_fetch_verifies_and_extracts_local_archive_idempotently(self):
        with tempfile.TemporaryDirectory(prefix="sigillum-corpus-") as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("fixture/score.musicxml", "<score-partwise/>")
            registry = self._registry(
                root,
                [
                    self._source(
                        "fixture",
                        archive,
                        digest=self._sha256(archive),
                        strip_components=1,
                    )
                ],
            )
            messages: list[str] = []
            fetcher = CorpusFetcher(registry, project_root=root, progress=messages.append)

            self.assertEqual(fetcher.fetch(["fixture"]), {"fixture": True})
            self.assertEqual(fetcher.fetch(["fixture"]), {"fixture": True})

            extracted = (
                root
                / "data"
                / "sources"
                / "fixture"
                / "source"
                / "dataset"
                / "score.musicxml"
            )
            self.assertEqual(extracted.read_text(encoding="utf-8"), "<score-partwise/>")
            state = fetcher.status()["sources"]["fixture"]
            self.assertEqual(state["status"], "complete")
            self.assertEqual(
                state["artifacts"]["dataset"]["digests"]["sha256"],
                self._sha256(archive),
            )
            self.assertTrue(
                any("extraction already verified" in message for message in messages)
            )

    def test_fetch_blocks_archive_traversal_without_writing_outside_target(self):
        with tempfile.TemporaryDirectory(prefix="sigillum-corpus-") as temporary:
            root = Path(temporary)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as target:
                target.writestr("../escaped.txt", "unsafe")
            registry = self._registry(
                root,
                [self._source("unsafe", archive, digest=self._sha256(archive))],
            )
            fetcher = CorpusFetcher(registry, project_root=root)

            self.assertEqual(fetcher.fetch(["unsafe"]), {"unsafe": False})

            self.assertFalse((root / "escaped.txt").exists())
            state = fetcher.status()["sources"]["unsafe"]
            self.assertEqual(state["status"], "blocked")
            self.assertIn(
                "unsafe archive path",
                state["artifacts"]["dataset"]["error"],
            )

    def test_fetch_continues_after_unavailable_source(self):
        with tempfile.TemporaryDirectory(prefix="sigillum-corpus-") as temporary:
            root = Path(temporary)
            available = root / "available.zip"
            with zipfile.ZipFile(available, "w") as target:
                target.writestr("score.musicxml", "<score-partwise/>")
            missing = root / "missing.zip"
            registry = self._registry(
                root,
                [
                    self._source("missing", missing),
                    self._source(
                        "available",
                        available,
                        digest=self._sha256(available),
                    ),
                ],
            )
            fetcher = CorpusFetcher(registry, project_root=root)

            self.assertEqual(
                fetcher.fetch(["missing", "available"]),
                {"missing": False, "available": True},
            )

            state = fetcher.status()["sources"]
            self.assertEqual(state["missing"]["status"], "blocked")
            self.assertEqual(state["available"]["status"], "complete")

    def test_registry_rejects_unknown_view_source(self):
        with tempfile.TemporaryDirectory(prefix="sigillum-corpus-") as temporary:
            root = Path(temporary)
            archive = root / "fixture.zip"
            with zipfile.ZipFile(archive, "w"):
                pass
            document = self._document([self._source("fixture", archive)])
            document["views"]["research_full"][0]["source_id"] = "unknown"
            path = root / "sources.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                CorpusRegistryError, "references unknown sources"
            ):
                CorpusRegistry.load(path)

    def _registry(
        self, root: Path, sources: list[dict[str, object]]
    ) -> CorpusRegistry:
        path = root / "sources.json"
        path.write_text(json.dumps(self._document(sources)), encoding="utf-8")
        return CorpusRegistry.load(path)

    @staticmethod
    def _document(sources: list[dict[str, object]]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "data_root": "data",
            "views": {
                "research_full": [
                    {"source_id": source["id"], "selection": "all"}
                    for source in sources
                ]
            },
            "unresolved_sources": [],
            "sources": sources,
        }

    @staticmethod
    def _source(
        source_id: str,
        archive: Path,
        *,
        digest: str | None = None,
        strip_components: int = 0,
    ) -> dict[str, object]:
        artifact: dict[str, object] = {
            "id": "dataset",
            "url": archive.as_uri(),
            "filename": archive.name,
            "extract_to": "dataset",
            "strip_components": strip_components,
        }
        if digest is not None:
            artifact["expected_digest"] = f"sha256:{digest}"
        return {
            "id": source_id,
            "title": source_id.title(),
            "homepage": "https://example.test/source",
            "roles": ["test_fixture"],
            "rights": {"tier": "test"},
            "version": {"label": "test"},
            "artifacts": [artifact],
        }

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
