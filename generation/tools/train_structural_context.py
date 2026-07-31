"""Train the frozen same-score structural-continuation experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from generation.composition.observation_dataset import (
    ObservationDataset,
    ObservationDatasetError,
)
from generation.composition.structural_context import (
    StructuralContextDatasetBuilder,
    StructuralContextError,
    StructuralContextSpec,
    StructuralContextTrainer,
)
from generation.project_paths import ROOT


DEFAULT_SPEC = (
    ROOT
    / "experiments"
    / "whole_score"
    / "structural_context_v1"
    / "experiment.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    return parser


def _project_path(relative: str | Path) -> Path:
    return ROOT.joinpath(*Path(relative).parts)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec = StructuralContextSpec.load(arguments.spec)
        dataset = ObservationDataset.load(
            _project_path(Path(spec.observation_manifest))
        )
        prepared = StructuralContextDatasetBuilder(spec, dataset).prepare()
        report = StructuralContextTrainer(spec, prepared).train(
            _project_path(Path(spec.output_root))
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (ObservationDatasetError, StructuralContextError) as error:
        print(f"structural-context error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
