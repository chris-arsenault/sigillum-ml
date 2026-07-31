"""Audit or train criterion-specific pairwise composition critics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generation.composition.critic_learning import (
    CriticCorpusIndex,
    CriticCorpusPreparer,
    CriticLearningError,
    CriticLearningSpec,
    PairwiseCriticTrainer,
    _write_json,
    build_critic_corpus_index,
    critic_readiness_without_index,
    load_critic_pairs,
    verify_critic_artifacts,
)

DEFAULT_SPEC = Path("experiments/whole_score/critic_v1/experiment.json")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Audit and train Partitura-evidence pairwise critics."
    )
    value.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help=f"experiment contract (default: {DEFAULT_SPEC})",
    )
    subcommands = value.add_subparsers(dest="command", required=True)
    index = subcommands.add_parser(
        "index",
        help="content-address one or more generated trajectory evidence sets",
    )
    index.add_argument(
        "--source",
        nargs=3,
        action="append",
        metavar=("TRAJECTORY", "REVIEWS", "PREFERENCES"),
        required=True,
        help="repeat for each generated evidence set; paths must be under outputs/",
    )
    subcommands.add_parser(
        "audit",
        help="report corpus readiness without fabricating missing preferences",
    )
    subcommands.add_parser(
        "train",
        help="train only when every frozen criterion gate is satisfied",
    )
    subcommands.add_parser(
        "verify",
        help="verify the pinned checkpoint and report independently",
    )
    return value


def main() -> int:
    arguments = parser().parse_args()
    root = Path.cwd().resolve()
    spec = CriticLearningSpec.load(root / arguments.spec)
    output_root = root.joinpath(*spec.output_root.parts)
    index_path = root.joinpath(*spec.corpus_index.parts)
    if arguments.command == "index":
        result = build_critic_corpus_index(
            project_root=root,
            sources=arguments.source,
            output_path=index_path,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "verify":
        result = verify_critic_artifacts(
            spec=spec,
            checkpoint_path=output_root / "checkpoint.pt",
            report_path=output_root / "report.json",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if not index_path.exists():
        report = critic_readiness_without_index(
            spec,
            reason=f"critic corpus index is absent: {spec.corpus_index.as_posix()}",
        )
        _write_json(output_root / "readiness.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2 if arguments.command == "train" else 0
    index = CriticCorpusIndex.load(index_path)
    pairs = load_critic_pairs(project_root=root, index=index)
    preparer = CriticCorpusPreparer(
        spec,
        pairs,
        corpus_index_digest=index.digest,
    )
    prepared = preparer.prepare(require_ready=arguments.command == "train")
    _write_json(output_root / "readiness.json", prepared.audit)
    if arguments.command == "audit":
        print(json.dumps(prepared.audit, indent=2, sort_keys=True))
        return 0
    report = PairwiseCriticTrainer(spec, prepared).train(output_root=output_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CriticLearningError as error:
        raise SystemExit(f"critic learning failed: {error}") from error
