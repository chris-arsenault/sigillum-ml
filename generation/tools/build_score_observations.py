"""Build and verify the Partitura-projected whole-score pilot dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from generation.composition.observation_dataset import (
    ObservationDataset,
    ObservationDatasetBuilder,
    ObservationDatasetError,
    ObservationDatasetSpec,
)
from generation.project_paths import ROOT

DEFAULT_SPEC = ROOT / "corpora" / "whole_score" / "pilot_v1.json"
PARTITURA = (
    ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"
).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "discover", help="validate selection, lineage, and split coverage"
    )

    build = commands.add_parser(
        "build", help="project scores and write the exact dataset manifest"
    )
    build.add_argument("--jobs", type=int, default=2)
    build.add_argument("--timeout-seconds", type=float, default=300.0)

    verify = commands.add_parser(
        "verify", help="verify the generated manifest and observations"
    )
    verify.add_argument("--manifest", type=Path)
    return parser


def _discovery(spec: ObservationDatasetSpec) -> dict[str, object]:
    targets = spec.discover(ROOT)
    return {
        "dataset_id": spec.dataset_id,
        "spec_digest": spec.digest,
        "score_count": len(targets),
        "source_score_counts": dict(
            sorted(Counter(target.collection.source_id for target in targets).items())
        ),
        "split_score_counts": dict(
            sorted(Counter(target.split for target in targets).items())
        ),
        "lineage_count": len({target.lineage.id for target in targets}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec = ObservationDatasetSpec.load(arguments.spec)
        if arguments.command == "discover":
            print(json.dumps(_discovery(spec), indent=2, sort_keys=True))
            return 0
        output_root = ROOT.joinpath(*spec.output_root.parts)
        if arguments.command == "verify":
            manifest = arguments.manifest or output_root / "manifest.json"
            dataset = ObservationDataset.load(manifest)
            observations = dataset.observations()
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "dataset_id": dataset.manifest["dataset_id"],
                        "ready": dataset.manifest["ready"],
                        "observations": len(observations),
                        "manifest_digest": dataset.manifest["manifest_digest"],
                    },
                    indent=2,
                )
            )
            return 0 if dataset.manifest["ready"] else 1
        builder = ObservationDatasetBuilder(
            spec,
            project_root=ROOT,
            partitura_bin=PARTITURA,
            progress=print,
            timeout_seconds=arguments.timeout_seconds,
        )
        manifest = builder.build(jobs=arguments.jobs)
        print(json.dumps(manifest["coverage"], indent=2, sort_keys=True))
        return 0 if manifest["ready"] else 1
    except ObservationDatasetError as error:
        print(f"dataset error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
