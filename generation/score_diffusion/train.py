"""Training loop for the whole-score self-supervised denoiser.

Plain DDPM training: sample windows from the ``train`` split, add noise at a random timestep,
regress the noise. Runs in the foreground; checkpoints and history land under gitignored
``outputs/``. Held-out reconstruction against the baselines is the actual success signal and is
computed by :mod:`generation.score_diffusion.evaluate`, invoked from the CLI after training.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import os
from pathlib import Path

# Keep ML jobs below half of the six-core development host by default. These are
# set before importing Torch so its BLAS/OpenMP pools inherit the same ceiling.
for _thread_env in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "2")

import torch

from generation.score_diffusion.dataset import WholeScoreRollDataset, iter_batches
from generation.score_diffusion.model import ScoreDenoiser, make_schedule, p_losses, count_parameters
from generation.score_diffusion import discrete as D


@dataclass
class TrainConfig:
    manifest: str = None
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-4
    width: int = 192
    depth: int = 8
    diffusion_steps: int = 200
    active_weight: float = 32.0
    x0_weight: float = 0.1
    min_active: int = 8
    seed: int = 0
    device: str = "cpu"
    cpu_threads: int = 2
    out_dir: str = "outputs/models/score_diffusion/fractal_denoise_v1"
    log_every: int = 50
    history: list = field(default_factory=list)


@torch.no_grad()
def eval_loss(model, dataset, sched, cfg, *, max_batches=20) -> float:
    model.eval()
    total, n = 0.0, 0
    g = torch.Generator().manual_seed(cfg.seed + 1)
    for bi, x0 in enumerate(iter_batches(dataset, cfg.batch_size, shuffle=True, generator=g)):
        if bi >= max_batches:
            break
        total += float(p_losses(model, x0.to(cfg.device), sched,
                               active_weight=cfg.active_weight, x0_weight=cfg.x0_weight))
        n += 1
    model.train()
    return total / max(1, n)


def train(cfg: TrainConfig):
    torch.set_num_threads(max(1, cfg.cpu_threads))
    torch.manual_seed(cfg.seed)
    train_ds = WholeScoreRollDataset(cfg.manifest, "train", min_active=cfg.min_active)
    val_ds = WholeScoreRollDataset(cfg.manifest, "validation", min_active=cfg.min_active)
    sched = make_schedule(cfg.diffusion_steps)
    model = ScoreDenoiser(width=cfg.width, depth=cfg.depth).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)

    print(f"train windows={len(train_ds)} val windows={len(val_ds)} params={count_parameters(model):,}")
    step = 0
    for epoch in range(cfg.epochs):
        running = 0.0
        nb = 0
        for x0 in iter_batches(train_ds, cfg.batch_size, shuffle=True, generator=gen):
            x0 = x0.to(cfg.device)
            loss = p_losses(model, x0, sched, active_weight=cfg.active_weight, x0_weight=cfg.x0_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach())
            nb += 1
            step += 1
            if step % cfg.log_every == 0:
                print(f"  epoch {epoch} step {step} loss {running / nb:.4f}")
        vloss = eval_loss(model, val_ds, sched, cfg)
        rec = {"epoch": epoch, "train_loss": running / max(1, nb), "val_loss": vloss}
        cfg.history.append(rec)
        print(f"epoch {epoch}: train {rec['train_loss']:.4f} val {vloss:.4f}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "model.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {k: v for k, v in asdict(cfg).items() if k != "history"},
                "history": cfg.history}, ckpt)
    print(f"saved checkpoint -> {ckpt}")
    return model, sched, cfg


def load_checkpoint(path, device="cpu"):
    blob = torch.load(path, map_location=device)
    c = blob["config"]
    model = ScoreDenoiser(width=c["width"], depth=c["depth"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    sched = make_schedule(c["diffusion_steps"])
    return model, sched, blob


@dataclass
class DiscreteTrainConfig:
    manifest: str = None
    epochs: int = 40
    batch_size: int = 16
    lr: float = 2e-4
    width: int = 192
    depth: int = 8
    diffusion_steps: int = 200
    pos_weight: float = 50.0
    min_active: int = 8
    seed: int = 0
    device: str = "cpu"
    cpu_threads: int = 2
    out_dir: str = "outputs/models/score_diffusion/fractal_denoise_v2"
    log_every: int = 50
    history: list = field(default_factory=list)


@torch.no_grad()
def eval_loss_discrete(model, dataset, sched, cfg, *, max_batches=20) -> float:
    model.eval()
    total, n = 0.0, 0
    g = torch.Generator().manual_seed(cfg.seed + 1)
    for bi, x0 in enumerate(iter_batches(dataset, cfg.batch_size, shuffle=True, generator=g)):
        if bi >= max_batches:
            break
        # x0 arrives in [-1,1]; discrete diffusion wants {0,1}.
        x0b = (x0 > 0).float().to(cfg.device)
        total += float(D.p_losses(model, x0b, sched, pos_weight=cfg.pos_weight))
        n += 1
    model.train()
    return total / max(1, n)


def train_discrete(cfg: DiscreteTrainConfig):
    torch.set_num_threads(max(1, cfg.cpu_threads))
    torch.manual_seed(cfg.seed)
    train_ds = WholeScoreRollDataset(cfg.manifest, "train", min_active=cfg.min_active)
    val_ds = WholeScoreRollDataset(cfg.manifest, "validation", min_active=cfg.min_active)
    sched = D.make_survival_schedule(cfg.diffusion_steps)
    model = D.OccupancyDenoiser(width=cfg.width, depth=cfg.depth).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)

    print(f"[discrete] train windows={len(train_ds)} val windows={len(val_ds)} "
          f"params={D.count_parameters(model):,}")
    step = 0
    for epoch in range(cfg.epochs):
        running, nb = 0.0, 0
        for x0 in iter_batches(train_ds, cfg.batch_size, shuffle=True, generator=gen):
            x0b = (x0 > 0).float().to(cfg.device)
            loss = D.p_losses(model, x0b, sched, pos_weight=cfg.pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += float(loss.detach())
            nb += 1
            step += 1
            if step % cfg.log_every == 0:
                print(f"  epoch {epoch} step {step} loss {running / nb:.4f}")
        vloss = eval_loss_discrete(model, val_ds, sched, cfg)
        rec = {"epoch": epoch, "train_loss": running / max(1, nb), "val_loss": vloss}
        cfg.history.append(rec)
        print(f"epoch {epoch}: train {rec['train_loss']:.4f} val {vloss:.4f}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "model.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {k: v for k, v in asdict(cfg).items() if k != "history"},
                "history": cfg.history}, ckpt)
    print(f"saved checkpoint -> {ckpt}")
    return model, sched, cfg
