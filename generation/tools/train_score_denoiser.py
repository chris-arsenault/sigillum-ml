"""Train + evaluate the DISCRETE (Bernoulli) whole-score denoiser (phase 11 / fractal_denoise_v2).

Phase 10 showed a continuous Gaussian DDPM is the wrong prior for a 99.8%-empty binary onset roll
(it decoded ~200x too dense and lost to persistence). This variant reformulates corruption and
generation as discrete "survival to silence / reveal from silence" over onset occupancy, trained
with class-balanced BCE. Everything else -- roll, lineage splits, baselines, honesty gates -- is
shared with phase 10, so the persistence F1 on the held-out ``test`` split is the same bar to clear.

Foreground, CPU-capped at <=3 threads:

    python -m generation.tools.train_score_denoiser --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from generation.project_paths import ROOT
from generation.score_diffusion import discrete as D
from generation.score_diffusion import evaluate as E
from generation.score_diffusion.dataset import WholeScoreRollDataset
from generation.score_diffusion.train import DiscreteTrainConfig, train_discrete

PILOT_MANIFEST = ROOT / "outputs" / "datasets" / "whole_score" / "pilot_v1" / "manifest.json"
DEFAULT_OUT = ROOT / "outputs" / "experiments" / "whole_score" / "fractal_denoise_v2"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=str, default=str(PILOT_MANIFEST))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--pos-weight", type=float, default=50.0)
    p.add_argument("--min-active", type=int, default=8)
    p.add_argument("--mask-measures", type=int, default=2)
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument("--decode-threshold", type=float, default=None,
                   help="Decode occupancy by this threshold instead of Bernoulli sampling "
                        "(calibrated decode for the ultra-sparse grid); omit for Bernoulli")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=2,
                   help="Torch CPU threads (default 2; must be <= 3 on this 6-core host)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return p


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        tmp = Path(handle.name)
    os.replace(tmp, path)


def _print_split(name: str, table: dict) -> None:
    print(f"\n== {name} (masked-measure active-cell F1) ==")
    for key in ("silence", "marginal", "persistence", "model"):
        r = table[key]
        print(f"  {key:12s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")
    ac = table["anti_collapse"]
    print(f"  beats_all_baselines={table['beats_all_baselines']}  "
          f"gen_density={ac['generated']['density_mean']:.4f} (auth {ac['authentic']['density_mean']:.4f})  "
          f"gen_repetition={ac['generated']['repetition_mean']:.3f} (auth {ac['authentic']['repetition_mean']:.3f})")


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.threads > 3:
        raise SystemExit("--threads must be <= 3 on this 6-core host (50% CPU ceiling)")
    torch.set_num_threads(max(1, args.threads))
    manifest = None if args.manifest in ("", str(PILOT_MANIFEST)) else args.manifest

    cfg = DiscreteTrainConfig(
        manifest=manifest, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        width=args.width, depth=args.depth, diffusion_steps=args.diffusion_steps,
        pos_weight=args.pos_weight, min_active=args.min_active, seed=args.seed,
        device=args.device, cpu_threads=args.threads,
        out_dir=str(Path("outputs/models/score_diffusion/fractal_denoise_v2")),
    )
    model, sched, cfg = train_discrete(cfg)

    report = {
        "experiment_id": "whole_score_fractal_denoise_v2",
        "formulation": "discrete_bernoulli_survival_reveal",
        "manifest": args.manifest,
        "params": D.count_parameters(model),
        "config": {k: getattr(cfg, k) for k in
                   ("epochs", "batch_size", "lr", "width", "depth", "diffusion_steps",
                    "pos_weight", "min_active", "seed", "cpu_threads")},
        "history": cfg.history,
        "mask_measures": args.mask_measures,
        "splits": {},
    }
    for split in ("validation", "test"):
        ds = WholeScoreRollDataset(manifest, split, min_active=args.min_active)
        g = torch.Generator().manual_seed(args.seed + 100)
        table = E.compare_to_baselines(
            ds, model, sched, mask_measures=args.mask_measures, limit=args.eval_limit,
            device=args.device, generator=g,
            reconstructor=D.discrete_reconstructor(model, sched, device=args.device, generator=g,
                                                   threshold=args.decode_threshold),
            sampler=D.discrete_sampler(model, sched, device=args.device, generator=g,
                                       threshold=args.decode_threshold))
        table["coverage"] = ds.coverage()
        report["splits"][split] = table
        _print_split(split, table)

    out_json = args.out / "report.json"
    _write_json(out_json, report)
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
