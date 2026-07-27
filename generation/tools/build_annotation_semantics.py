"""Build, verify, and baseline Partitura-bound score annotations."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from generation.composition.annotation_dataset import (
    AnnotationDataset,
    AnnotationDatasetBuilder,
    AnnotationDatasetError,
    AnnotationDatasetSpec,
)
from generation.composition.observation_dataset import ObservationDatasetError
from generation.composition.representation_baselines import (
    RepresentationBaselineRunner,
)
from generation.project_paths import ROOT

DEFAULT_SPEC = ROOT / "corpora" / "whole_score" / "annotation_semantics_v1.json"
PARTITURA = (
    ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"
).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", help="bind source annotations and publish the exact manifest"
    )
    build.add_argument("--jobs", type=int, default=2)
    build.add_argument("--timeout-seconds", type=float, default=300.0)
    verify = commands.add_parser(
        "verify", help="verify all annotation observations and examples"
    )
    verify.add_argument("--manifest", type=Path)
    baseline = commands.add_parser(
        "baseline", help="run split-safe majority and nearest-centroid baselines"
    )
    baseline.add_argument("--manifest", type=Path)
    baseline.add_argument("--output", type=Path)
    return parser


def _output_root(spec: AnnotationDatasetSpec) -> Path:
    return ROOT.joinpath(*spec.output_root.parts)


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
        json.dump(value, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        spec = AnnotationDatasetSpec.load(arguments.spec)
        output_root = _output_root(spec)
        if arguments.command == "build":
            manifest = AnnotationDatasetBuilder(
                spec,
                project_root=ROOT,
                partitura_bin=PARTITURA,
                timeout_seconds=arguments.timeout_seconds,
            ).build(jobs=arguments.jobs)
            print(json.dumps(manifest["coverage"], indent=2, sort_keys=True))
            return 0 if manifest["ready"] else 1
        manifest_path = arguments.manifest or output_root / "manifest.json"
        dataset = AnnotationDataset.load(manifest_path)
        if arguments.command == "verify":
            examples = dataset.examples()
            print(
                json.dumps(
                    {
                        "status": "ok",
                        "dataset_id": dataset.manifest["dataset_id"],
                        "ready": dataset.manifest["ready"],
                        "examples": len(examples),
                        "manifest_digest": dataset.manifest["manifest_digest"],
                    },
                    indent=2,
                )
            )
            return 0 if dataset.manifest["ready"] else 1
        report = RepresentationBaselineRunner(spec, dataset).run()
        report_path = arguments.output or output_root / "baselines.json"
        _write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (AnnotationDatasetError, ObservationDatasetError) as error:
        print(f"annotation dataset error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
