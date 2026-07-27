import json
import tempfile
import unittest
from pathlib import Path

from generation.composition.observation_dataset import (
    ObservationDataset,
    ObservationDatasetBuilder,
    ObservationDatasetError,
    ObservationDatasetSpec,
    ScoreObservation,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "corpora" / "whole_score" / "pilot_v1.json"
PARTITURA = ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"

MUSICXML = """\
<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <work><work-title>Dataset Fixture</work-title></work>
  <part-list>
    <score-part id="P1"><part-name>Flute</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <time><beats>4</beats><beat-type>4</beat-type></time>
      </attributes>
      <note>
        <pitch><step>C</step><octave>5</octave></pitch>
        <duration>4</duration>
      </note>
      <note><rest/><duration>12</duration></note>
    </measure>
  </part>
</score-partwise>
"""


class ObservationDatasetTests(unittest.TestCase):
    def test_project_spec_freezes_targets_and_lineage_split_policy(self):
        spec = ObservationDatasetSpec.load(SPEC)

        self.assertEqual(spec.dataset_id, "whole_score_observation_pilot_v1")
        self.assertEqual(
            sum(collection.expected_score_count for collection in spec.collections),
            110,
        )
        self.assertEqual(
            {collection.source_id for collection in spec.collections},
            {"openscore_orchestra", "s3_symbolic_symphonies"},
        )
        self.assertEqual(spec.split_policy["key"], "composition_lineage")
        self.assertIn(
            "one_composite_quality_score",
            spec.training_targets["excluded_targets"],
        )

    def test_builder_projects_scores_and_loader_verifies_split_views(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-observation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec_path = self._fixture(root)
            spec = ObservationDatasetSpec.load(spec_path)
            builder = ObservationDatasetBuilder(
                spec,
                project_root=root,
                partitura_bin=PARTITURA,
                timeout_seconds=30,
            )

            manifest = builder.build(jobs=2)

            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["coverage"]["score_count"], 2)
            self.assertEqual(
                manifest["coverage"]["split_score_counts"],
                {"test": 1, "train": 1},
            )
            self.assertEqual(
                manifest["coverage"]["scores_with_annotations"],
                2,
            )
            self.assertEqual(
                manifest["coverage"]["pitched_note_count"],
                2,
            )

            dataset = ObservationDataset.load(
                root / "outputs" / "pilot" / "manifest.json"
            )
            self.assertEqual(len(dataset.observations()), 2)
            self.assertEqual(len(dataset.observations(split="train")), 1)
            self.assertEqual(
                dataset.observations(split="test")[0].summary["event_count"],
                2,
            )

    def test_score_observation_rejects_digest_tampering(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-observation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec = ObservationDatasetSpec.load(self._fixture(root))
            manifest = ObservationDatasetBuilder(
                spec,
                project_root=root,
                partitura_bin=PARTITURA,
                timeout_seconds=30,
            ).build(jobs=1)
            observation_path = (
                root
                / "outputs"
                / "pilot"
                / manifest["records"][0]["observation_file"]
            )
            data = json.loads(observation_path.read_text(encoding="utf-8"))
            data["summary"]["event_count"] += 1

            with self.assertRaisesRegex(
                ObservationDatasetError, "observation digest mismatch"
            ):
                ScoreObservation.from_dict(data)

    def test_discovery_rejects_lineage_overlap(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-observation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec_path = self._fixture(root)
            data = json.loads(spec_path.read_text(encoding="utf-8"))
            data["collections"][0]["lineages"][1]["score_globs"] = [
                "works/**/*.musicxml"
            ]
            spec_path.write_text(json.dumps(data), encoding="utf-8")
            spec = ObservationDatasetSpec.load(spec_path)

            with self.assertRaisesRegex(
                ObservationDatasetError, "must match exactly one lineage"
            ):
                spec.discover(root)

    def _fixture(self, root: Path) -> Path:
        source = root / "data" / "sources" / "fixture"
        for work in ("a", "b"):
            directory = source / "works" / work
            directory.mkdir(parents=True)
            (directory / "score.musicxml").write_text(
                MUSICXML,
                encoding="utf-8",
            )
            (directory / "labels.csv").write_text(
                "onset,label\n0,theme\n",
                encoding="utf-8",
            )
        document = {
            "schema_version": 1,
            "dataset_id": "fixture_dataset",
            "observation_schema_version": 1,
            "data_root": "data",
            "output_root": "outputs/pilot",
            "training_targets": {
                "representation": ["structure"],
                "criterion_specific_critics": ["coherence"],
                "ruby_authoritative_auxiliaries": ["mechanical_validity"],
                "excluded_targets": ["composite_quality"],
            },
            "split_policy": {
                "key": "composition_lineage",
                "rule": "fixture",
            },
            "collections": [
                {
                    "id": "fixture_collection",
                    "source_id": "fixture_source",
                    "source_version": {"label": "fixture"},
                    "root": "sources/fixture",
                    "score_glob": "works/**/*.musicxml",
                    "exclude_globs": [],
                    "expected_score_count": 2,
                    "annotation_rules": [
                        {
                            "kind": "labels",
                            "patterns": ["{directory}/labels.csv"],
                        }
                    ],
                    "lineages": [
                        {
                            "id": "work_a",
                            "split": "train",
                            "score_globs": ["works/a/*.musicxml"],
                        },
                        {
                            "id": "work_b",
                            "split": "test",
                            "score_globs": ["works/b/*.musicxml"],
                        },
                    ],
                }
            ],
            "minimum_coverage": {
                "score_count": 2,
                "failed_score_count": 0,
                "source_score_counts": {"fixture_source": 2},
                "split_score_counts": {"train": 1, "test": 1},
                "lineage_count": 2,
                "scores_with_annotations": 2,
            },
        }
        spec_path = root / "spec.json"
        spec_path.write_text(json.dumps(document), encoding="utf-8")
        return spec_path


if __name__ == "__main__":
    unittest.main()
