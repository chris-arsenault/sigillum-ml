import json
import tempfile
import unittest
from pathlib import Path

from generation.composition.annotation_dataset import (
    AnnotationDataset,
    AnnotationDatasetBuilder,
    AnnotationDatasetError,
    AnnotationDatasetSpec,
    AnnotationObservation,
)
from generation.composition.observation_dataset import (
    ObservationDatasetBuilder,
    ObservationDatasetSpec,
)
from generation.composition.representation_baselines import (
    RepresentationBaselineRunner,
)

ROOT = Path(__file__).resolve().parents[1]
PARTITURA = ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"

MUSICXML = """\
<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>flute1</part-name></score-part>
    <score-part id="P2"><part-name>violin1</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="2"><note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration></note></measure>
    <measure number="3"><note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration></note></measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>1</divisions><time><beats>4</beats><beat-type>4</beat-type></time></attributes>
      <note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration></note>
    </measure>
    <measure number="2"><note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration></note></measure>
    <measure number="3"><note><pitch><step>E</step><octave>5</octave></pitch><duration>4</duration></note></measure>
  </part>
</score-partwise>
"""


class AnnotationDatasetTests(unittest.TestCase):
    def test_builder_binds_examples_and_baselines_are_split_safe(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-annotation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec = self._fixture(root)

            manifest = AnnotationDatasetBuilder(
                spec,
                project_root=root,
                partitura_bin=PARTITURA,
                timeout_seconds=30,
            ).build(jobs=2)

            self.assertTrue(manifest["ready"])
            self.assertEqual(manifest["coverage"]["score_count"], 3)
            self.assertEqual(manifest["coverage"]["binding_failure_count"], 0)
            self.assertEqual(manifest["coverage"]["failed_audit_count"], 0)
            self.assertEqual(manifest["coverage"]["warning_code_counts"], {})
            self.assertEqual(
                manifest["coverage"]["supported_targets_with_examples"], 4
            )
            dataset = AnnotationDataset.load(
                root / "outputs" / "annotations" / "manifest.json"
            )
            self.assertEqual(
                len(dataset.examples(target="prominent_part", split="train")), 2
            )
            report = RepresentationBaselineRunner(spec, dataset).run()
            self.assertTrue(
                all(
                    check["passed"]
                    for check in report["lineage_leakage_checks"]
                )
            )
            prominent = next(
                target
                for target in report["targets"]
                if target["target"] == "prominent_part"
            )
            self.assertEqual(
                set(prominent["models"]), {"majority", "nearest_centroid"}
            )
            self.assertEqual(
                prominent["models"]["majority"]["test"]["example_count"], 2
            )

    def test_annotation_observation_rejects_digest_tampering(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-annotation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            manifest = AnnotationDatasetBuilder(
                spec,
                project_root=root,
                partitura_bin=PARTITURA,
                timeout_seconds=30,
            ).build(jobs=1)
            path = (
                root
                / "outputs"
                / "annotations"
                / manifest["records"][0]["annotation_observation_file"]
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            document["summary"]["example_count"] += 1

            with self.assertRaisesRegex(
                AnnotationDatasetError, "annotation observation digest mismatch"
            ):
                AnnotationObservation.from_dict(document)

    def test_annotation_observation_rejects_stale_projector(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-annotation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            manifest = AnnotationDatasetBuilder(
                spec,
                project_root=root,
                partitura_bin=PARTITURA,
                timeout_seconds=30,
            ).build(jobs=1)
            path = (
                root
                / "outputs"
                / "annotations"
                / manifest["records"][0]["annotation_observation_file"]
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            document["projector"] = "partitura-annotation-observation-v0"

            with self.assertRaisesRegex(
                AnnotationDatasetError, "projector revision is unsupported"
            ):
                AnnotationObservation.from_dict(document)

    def test_spec_rejects_supported_target_without_baselines(self):
        with tempfile.TemporaryDirectory(
            prefix="sigillum-annotation-dataset-"
        ) as temporary:
            root = Path(temporary)
            spec = self._fixture(root)
            path = root / "annotation-spec.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["targets"][0]["baselines"] = []
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(
                AnnotationDatasetError, "require a metric and baselines"
            ):
                AnnotationDatasetSpec.load(path)

            self.assertEqual(spec.dataset_id, "fixture_annotation_dataset")

    def _fixture(self, root: Path) -> AnnotationDatasetSpec:
        observation_manifest = self._observation_fixture(root)
        targets = [
            self._target(target)
            for target in (
                "prominent_part",
                "structural_part_relation",
                "material_recurrence",
                "seam_boundary",
            )
        ]
        targets.append(
            {
                "id": "candidate_to_original_change",
                "kind": "representation",
                "representation_target": "candidate_to_original_change",
                "availability": "unavailable",
                "source": "trajectory",
                "meaning": "candidate change",
                "unit": "pair",
                "metric": None,
                "baselines": [],
                "reason": "fixture has no candidates",
            }
        )
        document = {
            "schema_version": 1,
            "dataset_id": "fixture_annotation_dataset",
            "annotation_schema_version": 1,
            "data_root": "data",
            "observation_manifest": "outputs/observations/manifest.json",
            "expected_observation_manifest_digest": observation_manifest[
                "manifest_digest"
            ],
            "output_root": "outputs/annotations",
            "profiles": [
                {
                    "collection_id": "fixture_collection",
                    "profile": "openscore_hauptstimme_v1",
                    "input_kinds": [
                        "hauptstimme_annotations",
                        "part_relations",
                    ],
                    "reference_only_kinds": [],
                }
            ],
            "targets": targets,
            "minimum_coverage": {
                "score_count": 3,
                "failed_score_count": 0,
                "binding_failure_count": 0,
                "failed_audit_count": 0,
                "profile_score_counts": {
                    "openscore_hauptstimme_v1": 3
                },
                "supported_targets_with_examples": 4,
                "unavailable_target_count": 1,
            },
        }
        path = root / "annotation-spec.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return AnnotationDatasetSpec.load(path)

    def _observation_fixture(self, root: Path):
        source = root / "data" / "sources" / "fixture"
        lineages = []
        for work, split in (
            ("a", "train"),
            ("b", "validation"),
            ("c", "test"),
        ):
            directory = source / "works" / work
            directory.mkdir(parents=True)
            (directory / "score.musicxml").write_text(
                MUSICXML, encoding="utf-8"
            )
            (directory / "annotations.csv").write_text(
                "qstamp,measure,beat,measure_fraction,label,part,part_num,instrument\n"
                "0,1,1,0,a,Flute,0,Flute\n"
                "4,2,1,0,a,Violin,1,Violin\n",
                encoding="utf-8",
            )
            (directory / "relations.csv").write_text(
                "qstamp_start,qstamp_end,flute1,violin1\n"
                "0,4,Main Part,U(Main)\n"
                "4,12,P5(Main),Main Part\n",
                encoding="utf-8",
            )
            lineages.append(
                {
                    "id": f"work_{work}",
                    "split": split,
                    "score_globs": [f"works/{work}/*.musicxml"],
                }
            )
        document = {
            "schema_version": 1,
            "dataset_id": "fixture_observation_dataset",
            "observation_schema_version": 1,
            "data_root": "data",
            "output_root": "outputs/observations",
            "training_targets": {
                "representation": ["structure"],
                "criterion_specific_critics": ["coherence"],
                "ruby_authoritative_auxiliaries": ["validity"],
                "excluded_targets": ["quality"],
            },
            "split_policy": {"key": "composition_lineage", "rule": "fixture"},
            "collections": [
                {
                    "id": "fixture_collection",
                    "source_id": "fixture_source",
                    "source_version": {"label": "fixture"},
                    "root": "sources/fixture",
                    "score_glob": "works/**/*.musicxml",
                    "exclude_globs": [],
                    "expected_score_count": 3,
                    "annotation_rules": [
                        {
                            "kind": "hauptstimme_annotations",
                            "patterns": ["{directory}/annotations.csv"],
                        },
                        {
                            "kind": "part_relations",
                            "patterns": ["{directory}/relations.csv"],
                        },
                    ],
                    "lineages": lineages,
                }
            ],
            "minimum_coverage": {
                "score_count": 3,
                "failed_score_count": 0,
                "source_score_counts": {"fixture_source": 3},
                "split_score_counts": {
                    "train": 1,
                    "validation": 1,
                    "test": 1,
                },
                "lineage_count": 3,
                "scores_with_annotations": 3,
            },
        }
        path = root / "observation-spec.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        spec = ObservationDatasetSpec.load(path)
        return ObservationDatasetBuilder(
            spec,
            project_root=root,
            partitura_bin=PARTITURA,
            timeout_seconds=30,
        ).build(jobs=2)

    @staticmethod
    def _target(target_id: str):
        return {
            "id": target_id,
            "kind": "representation",
            "representation_target": "structure",
            "availability": "supported",
            "source": "fixture",
            "meaning": target_id,
            "unit": "fixture",
            "metric": "macro_f1",
            "baselines": ["majority", "nearest_centroid"],
        }


if __name__ == "__main__":
    unittest.main()
