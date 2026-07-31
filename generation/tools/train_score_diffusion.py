"""Train + evaluate the full-corpus self-supervised whole-score denoiser (phase 10).

This is the first end-to-end test of the "start from noise, refine toward a plausible full
score" direction on the *whole* 110-score corpus with **no labels**. It trains a continuous DDPM
denoiser over a meter-normalized multi-family onset roll and then measures whether it can
reconstruct masked measures of held-out scores better than silence / marginal-frequency /
persistence baselines -- on both the ``validation`` split and the unseen-lineage ``test`` split --
without collapsing to silence or a single repeated column.

Checkpoints and the measured report land under gitignored ``outputs/``; the experiment contract
and the curated ``report.md`` stay in Git. Run in the foreground:

    python -m generation.tools.train_score_diffusion --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

from generation.project_paths import ROOT
from generation.score_diffusion import evaluate as E
from generation.score_diffusion.dataset import WholeScoreRollDataset
from generation.score_diffusion.model import count_parameters
from generation.score_diffusion.train import TrainConfig, train

PILOT_MANIFEST = ROOT / "outputs" / "datasets" / "whole_score" / "pilot_v1" / "manifest.json"
DEFAULT_OUT = ROOT / "outputs" / "experiments" / "whole_score" / "fractal_denoise_v1"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=str, default=str(PILOT_MANIFEST))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--active-weight", type=float, default=32.0)
    p.add_argument("--x0-weight", type=float, default=0.1)
    p.add_argument("--min-active", type=int, default=8)
    p.add_argument("--mask-measures", type=int, default=2)
    p.add_argument("--eval-limit", type=int, default=None,
                   help="cap windows per split during evaluation (sampling is slow on CPU)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--threads", type=int, default=2,
                   help="Torch CPU threads (default 2; keep at or below 3 on this 6-core host)")
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

    cfg = TrainConfig(
        manifest=manifest, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        width=args.width, depth=args.depth, diffusion_steps=args.diffusion_steps,
        active_weight=args.active_weight, x0_weight=args.x0_weight,
        min_active=args.min_active, seed=args.seed, device=args.device,
        cpu_threads=args.threads,
        out_dir=str(Path("outputs/models/score_diffusion/fractal_denoise_v1")),
    )
    model, sched, cfg = train(cfg)

    report = {
        "experiment_id": "whole_score_fractal_denoise_v1",
        "manifest": args.manifest,
        "params": count_parameters(model),
        "config": {k: getattr(cfg, k) for k in
                   ("epochs", "batch_size", "lr", "width", "depth", "diffusion_steps",
                    "active_weight", "x0_weight", "min_active", "seed", "cpu_threads")},
        "history": cfg.history,
        "mask_measures": args.mask_measures,
        "splits": {},
    }
    for split in ("validation", "test"):
        ds = WholeScoreRollDataset(manifest, split, min_active=args.min_active)
        g = torch.Generator().manual_seed(args.seed + 100)
        table = E.compare_to_baselines(
            ds, model, sched, mask_measures=args.mask_measures, limit=args.eval_limit,
            device=args.device, generator=g)
        table["coverage"] = ds.coverage()
        report["splits"][split] = table
        _print_split(split, table)

    out_json = args.out / "report.json"
    _write_json(out_json, report)
    print(f"\nwrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
