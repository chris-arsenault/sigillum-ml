"""Train or verify the frozen multi-task score-span encoder experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from generation.composition.annotation_dataset import (
    AnnotationDataset,
    AnnotationDatasetError,
)
from generation.composition.score_span_encoder import (
    ScoreSpanDatasetPreparer,
    ScoreSpanEncoderError,
    ScoreSpanEncoderSpec,
    ScoreSpanEncoderTrainer,
    load_baseline_report,
    verify_encoder_artifacts,
)
from generation.project_paths import ROOT

DEFAULT_SPEC = (
    ROOT
    / "experiments"
    / "whole_score"
    / "representation_v1"
    / "experiment.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="train, select, and evaluate the frozen pilot")
    commands.add_parser("verify", help="verify checkpoint and report lineage/digests")
    return parser


def _project_path(relative: Path) -> Path:
    return ROOT.joinpath(*relative.parts)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec = ScoreSpanEncoderSpec.load(arguments.spec)
        output_root = _project_path(Path(spec.output_root))
        if arguments.command == "verify":
            result = verify_encoder_artifacts(
                spec=spec,
                checkpoint_path=output_root / "checkpoint.pt",
                report_path=output_root / "report.json",
            )
        else:
            dataset = AnnotationDataset.load(
                _project_path(Path(spec.annotation_manifest))
            )
            prepared = ScoreSpanDatasetPreparer(spec, dataset).prepare()
            baseline = load_baseline_report(
                _project_path(Path(spec.baseline_report))
            )
            result = ScoreSpanEncoderTrainer(spec, prepared).train(
                output_root=output_root,
                baseline_report=baseline,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        AnnotationDatasetError,
        ScoreSpanEncoderError,
    ) as error:
        print(f"score-span encoder error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
