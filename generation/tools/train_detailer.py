"""Train the monophonic detailer (conditional DDPM, v2).

    python -m generation.tools.train_detailer --steps 8000

Coarse input = ornament-stripped structural line at flat velocity; target = the full detailed window
(key-normalised to C). Conditioned on density + character + harmony with classifier-free guidance.
Logs to sqlite MLflow (experiment `detailer`); saves to outputs/models/detailer/.
"""
import argparse
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import torch

from generation.fractal.dataset import melody_windows
from generation.fractal.model import Denoiser, make_schedule, p_losses
from generation.fractal.roll import FLAT_VEL, melody_to_roll, strip_ornaments

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "models" / "detailer"
MAXC, MAXH = 6, 4


def make_batch(windows, idxs, device):
    det, coa, char, harm, dens = [], [], [], [], []
    for i in idxs:
        w = windows[i]
        det.append(melody_to_roll(w.notes))
        coa.append(melody_to_roll(strip_ornaments(w.notes, w.figs), flat_vel=FLAT_VEL))
        char.append((w.character + [0] * MAXC)[:MAXC])
        harm.append((w.harmony + [0] * MAXH)[:MAXH])
        dens.append(w.density)
    return (torch.tensor(np.stack(det) * 2 - 1).to(device),
            torch.tensor(np.stack(coa) * 2 - 1).to(device),
            torch.tensor(char, device=device),
            torch.tensor(harm, device=device),
            torch.tensor(dens, dtype=torch.float32, device=device))


@torch.no_grad()
def evaluate(model, windows, sched, device, batch, rounds=8):
    model.eval()
    total = 0.0
    for _ in range(rounds):
        idx = [random.randrange(len(windows)) for _ in range(batch)]
        total += p_losses(model, *make_batch(windows, idx, device), sched, drop_p=0.0).item()
    model.train()
    return total / rounds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--diffusion-steps", type=int, default=200)
    p.add_argument("--drop-p", type=float, default=0.15)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed); random.seed(a.seed)
    device = "cpu"
    windows = melody_windows(); random.shuffle(windows)
    split = max(1, int(len(windows) * 0.97))
    train, val = windows[:split], windows[split:] or windows[-64:]
    sched = make_schedule(steps=a.diffusion_steps)
    model = Denoiser(width=a.width, depth=a.depth).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"windows: {len(windows)} ({len(train)}/{len(val)})  params: {params:.2f}M", flush=True)

    mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db"))
    mlflow.set_experiment("detailer")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = {"width": a.width, "depth": a.depth, "diffusion_steps": a.diffusion_steps}
    with mlflow.start_run(run_name="detailer_v2"):
        mlflow.log_params({**cfg, "batch": a.batch, "lr": a.lr, "steps": a.steps,
                           "n_windows": len(windows), "params_M": round(params, 3), "drop_p": a.drop_p})
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
        start = time.time(); best = float("inf")
        for step in range(1, a.steps + 1):
            idx = [random.randrange(len(train)) for _ in range(a.batch)]
            loss = p_losses(model, *make_batch(train, idx, device), sched, drop_p=a.drop_p)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % a.eval_every == 0 or step == 1:
                vloss = evaluate(model, val, sched, device, a.batch)
                mlflow.log_metrics({"train_loss": loss.item(), "val_loss": vloss}, step=step)
                print(f"step {step:5d}  train {loss.item():.4f}  val {vloss:.4f}  "
                      f"({(time.time()-start)/step*1000:.0f} ms/step)", flush=True)
                if vloss < best:
                    best = vloss
                    torch.save({"model": model.state_dict(), "config": cfg}, OUT_DIR / "model.pt")
        torch.save({"model": model.state_dict(), "config": cfg}, OUT_DIR / "model.pt")
        mlflow.log_metric("val_loss_final", vloss)
        print(f"saved to {OUT_DIR}  (best val {best:.4f})", flush=True)


if __name__ == "__main__":
    main()
