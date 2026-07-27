import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from generation.composition import BenchmarkError, BenchmarkManifest, EvaluationLab

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments" / "whole_score" / "v1_smoke" / "manifest.json"
FIXTURE = ROOT / "tests" / "fixtures" / "composition_kernel_study.rb"
PARTITURA = ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"


def evaluation_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            "-m",
            "generation.tools.evaluate_composition",
            *arguments,
        ),
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def partitura(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("ruby", str(PARTITURA), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


class EvaluationLabTests(unittest.TestCase):
    def test_frozen_manifest_covers_baselines_metrics_and_ablations(self):
        manifest = BenchmarkManifest.load(MANIFEST, root=ROOT)

        self.assertEqual(manifest.benchmark_id, "whole-score-v1-smoke")
        self.assertEqual(len(manifest.expected_cells()), 13)
        self.assertEqual(
            {item.strategy_id for item in manifest.strategies},
            {"one_shot", "fixed_agent_like", "deterministic_graph"},
        )
        self.assertEqual(len(manifest.ablations), 10)

        changed = manifest.to_dict()
        changed["title"] = "unfrozen mutation"
        with self.assertRaises(BenchmarkError):
            BenchmarkManifest.from_dict(changed)

    def test_cli_collects_runs_and_reports_held_out_blinded_comparison(self):
        with tempfile.TemporaryDirectory(prefix="sigillum-evaluation-") as temp:
            paths = self._paths(Path(temp))
            self._write_sources(paths)
            left = self._collect(paths, "one_shot", paths["left"], 1)
            right = self._collect(paths, "deterministic_graph", paths["right"], 3)
            public_review = self._review(paths, left["run_id"], right["run_id"])
            self._prefer(paths, public_review["review_id"])

            manifest = BenchmarkManifest.load(MANIFEST, root=ROOT)
            lab = EvaluationLab.from_jsonl(
                manifest, paths["runs"], paths["reviews"], paths["preferences"]
            )
            report = lab.report().to_dict()

            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["coverage"]["observed_runs"], 2)
            self.assertEqual(report["coverage"]["expected_runs"], 13)
            self.assertEqual(report["coverage"]["expected_human_comparisons"], 78)
            self.assertEqual(report["coverage"]["observed_human_comparisons"], 1)
            self.assertEqual(len(report["coverage"]["ready_review_assignments"]), 5)
            self.assertEqual(
                sum(
                    system["human_preference"]["coherence"]["wins"]
                    for system in report["systems"]
                ),
                1,
            )
            self.assertIn(
                "No composite score",
                lab.report().render_markdown(),
            )
            manifest_text = (Path(public_review["bundle"]) / "review.json").read_text(
                encoding="utf-8"
            )
            self.assertNotIn(left["run_id"], manifest_text)
            self.assertNotIn(right["run_id"], manifest_text)
            review_record = json.loads(paths["reviews"].read_text(encoding="utf-8"))
            review_record["subjects"][0]["source_digest"] = "sha256:" + "0" * 64
            paths["reviews"].write_text(
                json.dumps(review_record) + "\n", encoding="utf-8"
            )
            with self.assertRaises(BenchmarkError):
                EvaluationLab.from_jsonl(
                    manifest,
                    paths["runs"],
                    paths["reviews"],
                    paths["preferences"],
                )

    def _paths(self, directory: Path) -> dict[str, Path]:
        return {
            "left": directory / "left.rb",
            "right": directory / "right.rb",
            "runs": directory / "runs.jsonl",
            "reviews": directory / "reviews.jsonl",
            "preferences": directory / "preferences.jsonl",
            "bundles": directory / "bundles",
        }

    def _write_sources(self, paths: dict[str, Path]) -> None:
        shutil.copyfile(FIXTURE, paths["left"])
        source = FIXTURE.read_text(encoding="utf-8")
        variant = source.replace(
            'pitch_bars "C5 E5 | G5 E5"',
            'pitch_bars "C5 F5 | G5 E5"',
        )
        paths["right"].write_text(variant, encoding="utf-8")

    def _collect(
        self, paths: dict[str, Path], strategy: str, source: Path, model_calls: int
    ) -> dict[str, object]:
        result = evaluation_cli(
            "collect",
            str(MANIFEST),
            "--runs",
            str(paths["runs"]),
            "--source",
            str(source),
            "--case",
            "kernel-study",
            "--strategy",
            strategy,
            "--seed",
            "1729",
            "--candidate-count",
            str(model_calls),
            "--mechanically-valid-candidates",
            str(model_calls),
            "--accepted-edits",
            "1",
            "--model-calls",
            str(model_calls),
            "--wall-seconds",
            "0.5",
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def _review(
        self, paths: dict[str, Path], left_run: object, right_run: object
    ) -> dict[str, object]:
        result = partitura(
            "benchmark-review",
            str(paths["left"]),
            str(paths["right"]),
            "--left-run",
            str(left_run),
            "--right-run",
            str(right_run),
            "--benchmark",
            "whole-score-v1-smoke",
            "--case",
            "kernel-study",
            "--criterion",
            "coherence",
            "--reviews",
            str(paths["reviews"]),
            "--output",
            str(paths["bundles"]),
            "--seed",
            "evaluation-test",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def _prefer(self, paths: dict[str, Path], review_id: object) -> None:
        result = partitura(
            "benchmark-preference",
            "--reviews",
            str(paths["reviews"]),
            "--preferences",
            str(paths["preferences"]),
            "--review",
            str(review_id),
            "--outcome",
            "a",
            "--rater",
            "rater:evaluation-test",
            "--reason",
            "A has the more coherent four-bar arc.",
            "--confidence",
            "0.8",
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
