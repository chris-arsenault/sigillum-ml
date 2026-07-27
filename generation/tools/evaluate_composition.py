"""Operate the frozen whole-score evaluation lab."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

from generation.composition.benchmark import (
    BenchmarkCell,
    BenchmarkError,
    BenchmarkManifest,
)
from generation.composition.evaluation import EvaluationLab
from generation.composition.evaluation_store import (
    EvaluationRunStore,
    trajectory_effort,
)
from generation.composition.evaluation_run import EvaluationRun, RunEffort
from generation.project_paths import ROOT

PARTITURA = (
    ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"
).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="verify manifest and frozen inputs")
    verify.add_argument("manifest", type=Path)

    collect = commands.add_parser(
        "collect", help="measure and append one completed benchmark run"
    )
    collect.add_argument("manifest", type=Path)
    collect.add_argument("--runs", type=Path, required=True)
    collect.add_argument("--source", type=Path, required=True)
    collect.add_argument("--case", required=True)
    collect.add_argument("--strategy", required=True)
    collect.add_argument("--ablation", default="control")
    collect.add_argument("--seed", type=int, required=True)
    collect.add_argument("--trajectory", type=Path)
    collect.add_argument("--candidate-count", type=int)
    collect.add_argument("--mechanically-valid-candidates", type=int)
    collect.add_argument("--accepted-edits", type=int)
    collect.add_argument("--model-calls", type=int, required=True)
    collect.add_argument("--wall-seconds", type=float, required=True)

    report = commands.add_parser("report", help="aggregate runs and held-out reviews")
    report.add_argument("manifest", type=Path)
    report.add_argument("--runs", type=Path, required=True)
    report.add_argument("--reviews", type=Path)
    report.add_argument("--preferences", type=Path)
    report.add_argument("--json", action="store_true")
    return parser


def _manifest(path: Path) -> BenchmarkManifest:
    return BenchmarkManifest.load(path, root=ROOT)


def _measure(source: Path) -> dict[str, Any]:
    if not PARTITURA.is_file():
        raise BenchmarkError(f"Partitura executable does not exist: {PARTITURA}")
    command = subprocess.run(
        ("ruby", str(PARTITURA), "benchmark-score", str(source.resolve())),
        check=False,
        capture_output=True,
        text=True,
    )
    if command.returncode != 0:
        detail = command.stderr.strip() or command.stdout.strip() or "unknown error"
        raise BenchmarkError(f"Partitura measurement failed: {detail}")
    try:
        result = json.loads(command.stdout)
    except json.JSONDecodeError as error:
        raise BenchmarkError("Partitura measurement returned invalid JSON") from error
    if not isinstance(result, dict):
        raise BenchmarkError("Partitura measurement must return an object")
    return result


def _effort(arguments: argparse.Namespace) -> RunEffort:
    if arguments.trajectory:
        return trajectory_effort(
            arguments.trajectory,
            model_call_count=arguments.model_calls,
            wall_seconds=arguments.wall_seconds,
        )
    values = (
        arguments.candidate_count,
        arguments.mechanically_valid_candidates,
        arguments.accepted_edits,
    )
    if any(value is None for value in values):
        raise BenchmarkError(
            "collect without --trajectory requires --candidate-count, "
            "--mechanically-valid-candidates, and --accepted-edits"
        )
    return RunEffort.from_dict(
        {
            "candidate_count": arguments.candidate_count,
            "mechanically_valid_candidate_count": (
                arguments.mechanically_valid_candidates
            ),
            "accepted_edit_count": arguments.accepted_edits,
            "model_call_count": arguments.model_calls,
            "wall_seconds": arguments.wall_seconds,
        }
    )


def _collect(arguments: argparse.Namespace) -> dict[str, Any]:
    manifest = _manifest(arguments.manifest)
    cell = BenchmarkCell(
        case_id=arguments.case,
        strategy_id=arguments.strategy,
        ablation_id=arguments.ablation,
        seed=arguments.seed,
    )
    if cell.key() not in {item.key() for item in manifest.expected_cells()}:
        raise BenchmarkError(f"run targets an unexpected benchmark cell: {cell.key()}")
    run = EvaluationRun.create(
        manifest=manifest,
        cell=cell,
        measurement=_measure(arguments.source),
        effort=_effort(arguments),
    )
    EvaluationRunStore(arguments.runs).append(run)
    return run.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "verify":
            manifest = _manifest(arguments.manifest)
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "benchmark_id": manifest.benchmark_id,
                        "manifest_digest": manifest.manifest_digest,
                        "expected_runs": len(manifest.expected_cells()),
                    },
                    indent=2,
                )
            )
        elif arguments.command == "collect":
            print(json.dumps(_collect(arguments), indent=2))
        else:
            manifest = _manifest(arguments.manifest)
            lab = EvaluationLab.from_jsonl(
                manifest,
                arguments.runs,
                arguments.reviews,
                arguments.preferences,
            )
            report = lab.report()
            print(
                json.dumps(report.to_dict(), indent=2)
                if arguments.json
                else report.render_markdown()
            )
    except BenchmarkError as error:
        print(json.dumps({"status": "error", "message": str(error)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
