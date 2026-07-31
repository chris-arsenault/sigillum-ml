"""Train and evaluate the fractal harmonic refinement operator.

This is the first empirical test of the fractalization hypothesis on real corpus
harmony: can one shared learned operator invert a coarse functional-harmony
skeleton into a finer per-bar progression, better than copy/n-gram baselines,
while holding the coarse skeleton fixed?

Two honest splits are supported:

* ``composer_holdout`` uses the frozen dataset splits (train Mozart/Tchaikovsky,
  validate Dvorak, test Beethoven) -- cross-composer generalization.
* ``movement_holdout`` lets every composer appear in train and holds out one
  movement per composer -- seen-composer/unseen-movement.

Generated checkpoints, reports, and refined artifacts stay under ignored
``outputs/``. The experiment contract and measured report stay in Git.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from generation.fractal_score import (
    Bigram,
    BigramLM,
    CopyNearestPillar,
    HarmonyVocab,
    RefinementSchedule,
    TrainConfig,
    Unigram,
    build_progressions,
    evaluate_generative,
    evaluate_realism,
    evaluate_recursive,
    evaluate_steps,
    extract_windows,
    recursive_refine,
    split_windows,
    train_operator,
)
from generation.fractal_score.dataset import movement_holdout
from generation.fractal_score.harmony import NONE_TOKEN
from generation.fractal_score.ladder import revealed_positions
from generation.fractal_score.model import RefinementConfig
from generation.project_paths import ROOT

ANNOTATION_MANIFEST = (
    ROOT / "outputs" / "datasets" / "whole_score" / "annotation_semantics_v1" / "manifest.json"
)
PILOT_MANIFEST = (
    ROOT / "outputs" / "datasets" / "whole_score" / "pilot_v1" / "manifest.json"
)
DEFAULT_OUT = ROOT / "outputs" / "experiments" / "whole_score" / "fractal_harmony_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-mode",
        choices=["composer_holdout", "movement_holdout"],
        default="movement_holdout",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 13, 21])
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--coarsest", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=6)
    return parser


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _manifest_digest(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8")).get("manifest_digest", ""))


def _run_seed(
    progressions,
    schedule: RefinementSchedule,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], Any, HarmonyVocab, dict[str, list]]:
    if args.split_mode == "movement_holdout":
        progressions = movement_holdout(progressions, seed=seed)
    train = [p for p in progressions if p.split == "train"]
    vocab = HarmonyVocab.build(train, max_tokens=args.max_tokens)
    windows = extract_windows(
        progressions, vocab, window=args.window, stride=args.stride
    )
    grouped = split_windows(windows)
    model_config = RefinementConfig(
        vocab_size=len(vocab),
        key_count=vocab.key_count,
        level_count=schedule.level_count,
        max_length=args.window,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
    )
    train_config = TrainConfig(epochs=args.epochs, seed=seed)
    model, history = train_operator(
        grouped["train"], schedule, vocab, model_config, train_config
    )
    ignore = [vocab.pad_id, vocab.mask_id]
    baselines = [
        CopyNearestPillar(),
        Unigram.fit(grouped["train"], ignore),
        Bigram.fit(grouped["train"], ignore),
    ]
    bigram_lm = BigramLM.fit(grouped["train"], len(vocab), ignore)
    report: dict[str, Any] = {
        "seed": seed,
        "final_train_loss": history[-1]["loss"],
        "splits": {name: len(items) for name, items in grouped.items()},
        "evaluation": {},
    }
    for split in ("validation", "test"):
        if not grouped[split]:
            continue
        steps = evaluate_steps(model, vocab, grouped[split], schedule, baselines)
        recursive = evaluate_recursive(model, vocab, grouped[split], schedule, baselines)
        generative = evaluate_generative(model, vocab, grouped[split], schedule, bigram_lm)
        realism = evaluate_realism(model, vocab, grouped[split], schedule, baselines)
        report["evaluation"][split] = {
            "steps": [
                {
                    "child_stride": step.child_stride,
                    "positions": step.positions,
                    "learned_accuracy": step.learned_accuracy,
                    "baseline_accuracy": step.baseline_accuracy,
                }
                for step in steps
            ],
            "recursive": {
                "positions": recursive.positions,
                "learned_accuracy": recursive.learned_accuracy,
                "baseline_accuracy": recursive.baseline_accuracy,
                "parent_preserved": recursive.parent_preserved,
            },
            "generative_nll": {
                "positions": generative.positions,
                "learned": generative.learned_nll,
                "bigram_lm": generative.bigram_nll,
            },
            "realism": {
                "authentic_repeat_rate": realism.authentic_repeat_rate,
                "learned_repeat_rate": realism.learned_repeat_rate,
                "baseline_repeat_rate": realism.baseline_repeat_rate,
                "learned_js": realism.learned_js,
                "baseline_js": realism.baseline_js,
            },
        }
    return report, model, vocab, grouped


def _refined_artifacts(
    progressions,
    vocab: HarmonyVocab,
    model,
    schedule: RefinementSchedule,
    split: str,
    window: int,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for progression in progressions:
        if progression.split != split:
            continue
        ids = np.array(
            [vocab.encode(token) for token in progression.tokens], dtype=np.int64
        )
        if ids.shape[0] < schedule.coarsest * 2:
            continue
        key_id = vocab.encode_key(progression.home_key)
        # Refine window by window: the operator's positional length equals the
        # training window, so a long movement is engraved in aligned segments.
        forbid = (vocab.pad_id, vocab.mask_id, vocab.none_id, vocab.rare_id)
        generated = np.empty_like(ids)
        for start in range(0, ids.shape[0], window):
            segment = ids[start : start + window]
            generated[start : start + segment.shape[0]] = recursive_refine(
                model, vocab, segment, key_id, schedule, forbid_ids=forbid
            )
        pillars = set(revealed_positions(ids.shape[0], schedule.coarsest))
        bars = []
        for index, token in enumerate(progression.tokens):
            bars.append(
                {
                    "measure": progression.first_measure + index,
                    "is_pillar": index in pillars,
                    "authentic": token,
                    "model_refined": vocab.decode(int(generated[index])),
                }
            )
        artifacts.append(
            {
                "score_id": progression.score_id,
                "lineage_id": progression.lineage_id,
                "home_key": progression.home_key,
                "coarsest_stride": schedule.coarsest,
                "measures": len(progression.tokens),
                "bars": bars,
            }
        )
    return artifacts


def _aggregate(reports: list[dict[str, Any]], split: str) -> dict[str, Any]:
    with_split = [r for r in reports if split in r["evaluation"]]
    if not with_split:
        return {}
    child_strides = [
        step["child_stride"] for step in with_split[0]["evaluation"][split]["steps"]
    ]
    steps_summary = []
    for position, child in enumerate(child_strides):
        learned = [r["evaluation"][split]["steps"][position]["learned_accuracy"] for r in with_split]
        baseline_names = with_split[0]["evaluation"][split]["steps"][position][
            "baseline_accuracy"
        ].keys()
        baselines = {
            name: mean(
                r["evaluation"][split]["steps"][position]["baseline_accuracy"][name]
                for r in with_split
            )
            for name in baseline_names
        }
        best_baseline = max(baselines.values())
        steps_summary.append(
            {
                "child_stride": child,
                "learned_accuracy_mean": mean(learned),
                "learned_accuracy_std": pstdev(learned) if len(learned) > 1 else 0.0,
                "baseline_accuracy_mean": baselines,
                "learned_minus_best_baseline": mean(learned) - best_baseline,
            }
        )
    recursive = [r["evaluation"][split]["recursive"]["learned_accuracy"] for r in with_split]
    nll_learned = [r["evaluation"][split]["generative_nll"]["learned"] for r in with_split]
    nll_bigram = [r["evaluation"][split]["generative_nll"]["bigram_lm"] for r in with_split]
    return {
        "seeds": [r["seed"] for r in with_split],
        "steps": steps_summary,
        "recursive_learned_mean": mean(recursive),
        "generative_nll_learned_mean": mean(nll_learned),
        "generative_nll_bigram_mean": mean(nll_bigram),
    }


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")
    args = _parser().parse_args(argv)
    torch.set_num_threads(args.threads)
    schedule = RefinementSchedule.geometric(args.coarsest)
    base_progressions = build_progressions(ANNOTATION_MANIFEST, PILOT_MANIFEST)
    lineage_counts = Counter(p.lineage_id for p in base_progressions)

    seed_reports: list[dict[str, Any]] = []
    best_seed_state = None
    for seed in args.seeds:
        report, model, vocab, grouped = _run_seed(
            list(base_progressions), schedule, args, seed
        )
        seed_reports.append(report)
        print(
            f"seed {seed}: train_loss={report['final_train_loss']:.3f} "
            f"splits={report['splits']}"
        )
        if best_seed_state is None:
            best_seed_state = (seed, model, vocab)

    # Refined artifacts come from the first seed's split assignment and model.
    seed, model, vocab = best_seed_state
    if args.split_mode == "movement_holdout":
        artifact_progressions = movement_holdout(list(base_progressions), seed=seed)
    else:
        artifact_progressions = list(base_progressions)
    artifacts = _refined_artifacts(
        artifact_progressions, vocab, model, schedule, split="test", window=args.window
    )

    summary = {
        "experiment_id": "whole_score_fractal_harmony_v1",
        "split_mode": args.split_mode,
        "schedule": list(schedule.strides),
        "hyperparameters": {
            "seeds": args.seeds,
            "epochs": args.epochs,
            "window": args.window,
            "stride": args.stride,
            "coarsest": args.coarsest,
            "max_tokens": args.max_tokens,
            "d_model": args.d_model,
            "layers": args.layers,
            "heads": args.heads,
        },
        "provenance": {
            "annotation_manifest_digest": _manifest_digest(ANNOTATION_MANIFEST),
            "pilot_manifest_digest": _manifest_digest(PILOT_MANIFEST),
            "lineages": dict(sorted(lineage_counts.items())),
            "vocab_size": len(vocab),
        },
        "seed_reports": seed_reports,
        "aggregate": {
            "validation": _aggregate(seed_reports, "validation"),
            "test": _aggregate(seed_reports, "test"),
        },
    }
    out = args.out
    _write_json(out / f"report_{args.split_mode}.json", summary)
    _write_json(out / f"refined_examples_{args.split_mode}.json", artifacts)
    _save_checkpoint(
        out / f"checkpoint_{args.split_mode}.pt",
        model=model,
        vocab=vocab,
        schedule=schedule,
        seed=seed,
        split_mode=args.split_mode,
    )
    _print_headline(summary)
    return 0


def _save_checkpoint(
    path: Path,
    *,
    model,
    vocab: HarmonyVocab,
    schedule: RefinementSchedule,
    seed: int,
    split_mode: str,
) -> None:
    """Persist the first-seed operator so a refined artifact can be reproduced.

    Checkpoints live under ignored ``outputs/``; they are regeneratable, not a
    durable Git artifact.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "vocab": vocab.to_dict(),
            "schedule": list(schedule.strides),
            "seed": seed,
            "split_mode": split_mode,
        },
        path,
    )


def _print_headline(summary: dict[str, Any]) -> None:
    print(f"\n== fractal harmony ({summary['split_mode']}) ==")
    for split in ("validation", "test"):
        aggregate = summary["aggregate"].get(split)
        if not aggregate:
            continue
        print(f"[{split}] seeds={aggregate['seeds']}")
        for step in aggregate["steps"]:
            print(
                f"  child_stride={step['child_stride']:>2} "
                f"learned={step['learned_accuracy_mean']:.3f}"
                f"±{step['learned_accuracy_std']:.3f} "
                f"delta_vs_best_baseline={step['learned_minus_best_baseline']:+.3f} "
                f"baselines={ {k: round(v, 3) for k, v in step['baseline_accuracy_mean'].items()} }"
            )
        print(
            f"  recursive_learned={aggregate['recursive_learned_mean']:.3f} "
            f"nll_learned={aggregate['generative_nll_learned_mean']:.3f} "
            f"nll_bigram={aggregate['generative_nll_bigram_mean']:.3f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
