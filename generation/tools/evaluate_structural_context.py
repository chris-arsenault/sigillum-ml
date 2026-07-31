"""Evaluate the selected structural-context checkpoint on an external holdout."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from generation.composition.observation_dataset import (
    ObservationDataset,
    ObservationDatasetError,
)
from generation.composition.structural_context import (
    StructuralContextError,
    StructuralContextSpec,
    evaluate_structural_context_checkpoint,
)
from generation.project_paths import ROOT


DEFAULT_SPEC = (
    ROOT
    / "experiments"
    / "whole_score"
    / "structural_context_v4"
    / "experiment.json"
)
DEFAULT_CHECKPOINT = (
    ROOT
    / "outputs"
    / "experiments"
    / "whole_score"
    / "structural_context_v4"
    / "checkpoint.pt"
)
DEFAULT_HOLDOUT = (
    ROOT
    / "outputs"
    / "datasets"
    / "whole_score"
    / "structural_context_external_holdout_v1"
    / "manifest.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "experiments"
    / "whole_score"
    / "structural_context_v4"
    / "external_holdout_report.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


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
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec = StructuralContextSpec.load(arguments.spec)
        training_dataset = ObservationDataset.load(
            ROOT.joinpath(*spec.observation_manifest.parts)
        )
        holdout_dataset = ObservationDataset.load(arguments.holdout)
        report = evaluate_structural_context_checkpoint(
            spec=spec,
            checkpoint_path=arguments.checkpoint,
            training_dataset=training_dataset,
            holdout_dataset=holdout_dataset,
        )
        _write_json(arguments.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ObservationDatasetError, StructuralContextError) as error:
        print(f"structural-context evaluation error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
