"""Train the kernel-infilling encoder-decoder.

    python -m generation.tools.build_theme_dataset            # dataset first
    python -m generation.tools.train_infill --steps 2000

Spans are re-masked every batch (augmentation), so the model sees many kernels per melody. Loss is
on the decoder's gap fills (kind + event fields). Saves to outputs/models/theme_nn_infill/.
"""
import argparse
import random
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

from generation.theme_nn.dataset import DEFAULT_OUT
from generation.theme_nn.infill import CPOS_SIZE, DSTEP_SIZE, FIELDS, IKIND, N_FIG, N_MOTIF, InfillVocab
from generation.theme_nn.model_infill import FactoredEncDec, InfillConfig
from generation.theme_nn.vocab import load_examples

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models" / "theme_nn_infill"
FZ = {f: i for i, f in enumerate(FIELDS)}
NF = len(FIELDS)
# Étude-head loss weights (figuration / motif / chord-position) — regularisers that shape the trunk.
# The pitch heads (dstep, alt) are primary, full-weight, not listed here. Swappable.
AUX_W = {"fig": 0.3, "motif": 0.3, "cpos": 0.3}


def make_batch(examples, idxs, vocab, rng, enc_block, dec_block, keep, device):
    E = torch.zeros((len(idxs), enc_block, NF), dtype=torch.long)
    D = torch.zeros((len(idxs), dec_block, NF), dtype=torch.long)
    A = torch.zeros((len(idxs), dec_block, 4), dtype=torch.long)   # (fig, motif, cpos, dstep) targets
    epad = torch.ones((len(idxs), enc_block), dtype=torch.bool)
    dpad = torch.ones((len(idxs), dec_block), dtype=torch.bool)
    for b, i in enumerate(idxs):
        pair = vocab.encode_pair(examples[i], rng, keep)
        if pair is None:
            continue
        enc, dec, aux = pair
        enc, dec, aux = enc[:enc_block], dec[:dec_block], aux[:dec_block]
        for t, p in enumerate(enc):
            E[b, t] = torch.tensor(p); epad[b, t] = False
        for t, (p, a) in enumerate(zip(dec, aux)):
            D[b, t] = torch.tensor(p); A[b, t] = torch.tensor(a); dpad[b, t] = False
    return E.to(device), D.to(device), A.to(device), epad.to(device), dpad.to(device)


def loss_fn(model, E, D, A, epad, dpad):
    logits = model(E, D, enc_pad=epad, dec_pad=dpad)
    target = D[:, 1:]
    aux = A[:, 1:]
    real = ~dpad[:, 1:]                                  # non-pad decoder targets
    tgt_kind = target[:, :, FZ["kind"]]
    event = real & (tgt_kind == IKIND["EVENT"])
    losses = {"kind": F.cross_entropy(logits["kind"][:, :-1][real], tgt_kind[real])}
    losses["dstep"] = F.cross_entropy(logits["dstep"][:, :-1][event], aux[:, :, 3][event])  # primary pitch move
    losses["alt"] = F.cross_entropy(logits["alt"][:, :-1][event], target[:, :, FZ["alt"]][event])  # accidental
    losses["dur"] = F.cross_entropy(logits["dur"][:, :-1][event], target[:, :, FZ["dur"]][event])
    for j, f in ((0, "fig"), (1, "motif"), (2, "cpos")):                                    # étude heads
        losses[f] = AUX_W[f] * F.cross_entropy(logits[f][:, :-1][event], aux[:, :, j][event])
    return sum(losses.values()), losses


CORE = ("kind", "dstep", "alt", "dur")   # the predicted fields: kind + diatonic step + accidental + duration


@torch.no_grad()
def evaluate(model, examples, vocab, rng, enc_block, dec_block, keep, device, batch, rounds=8):
    model.eval()
    total = core = 0.0
    for _ in range(rounds):
        idx = [rng.randrange(len(examples)) for _ in range(batch)]
        loss, parts = loss_fn(model, *make_batch(examples, idx, vocab, rng, enc_block, dec_block, keep, device))
        total += loss.item()
        core += sum(parts[f].item() for f in CORE)      # exclude the étude heads
    model.train()
    return total / rounds, core / rounds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DEFAULT_OUT))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--enc-block", type=int, default=384)
    p.add_argument("--dec-block", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--keep", type=float, default=0.4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    device = "cpu"
    examples = [e for e in load_examples(Path(a.data)) if len(e["events"]) >= 6]
    rng.shuffle(examples)
    split = max(1, int(len(examples) * 0.95))
    train, val = examples[:split], examples[split:] or examples[-8:]
    vocab = InfillVocab.from_examples(examples)
    cfg = InfillConfig(field_sizes=vocab.field_sizes(), enc_block=a.enc_block, dec_block=a.dec_block,
                       n_layer=a.layers, n_head=a.heads, d_model=a.d_model,
                       n_fig=N_FIG, n_motif=N_MOTIF, n_cpos=CPOS_SIZE, n_dstep=DSTEP_SIZE)
    model = FactoredEncDec(cfg).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"examples: {len(examples)} ({len(train)}/{len(val)})  field_sizes: {vocab.field_sizes()}  "
          f"params: {params:.1f}M  étude: fig={N_FIG} motif={N_MOTIF} (w={AUX_W})")

    # Pin the sqlite backend explicitly — mlflow 3.13's file store is in maintenance mode and the
    # bare default now raises, which would silently drop a run's metrics.
    mlflow.set_tracking_uri("sqlite:///" + str(Path(__file__).resolve().parents[2] / "mlflow.db"))
    mlflow.set_experiment("theme_nn")
    with mlflow.start_run(run_name="infill_etude"):
        mlflow.log_params({"variant": "infill_encdec", "d_model": a.d_model, "layers": a.layers,
                           "heads": a.heads, "enc_block": a.enc_block, "dec_block": a.dec_block,
                           "batch": a.batch, "lr": a.lr, "steps": a.steps, "keep": a.keep,
                           "params_M": round(params, 2), "aux_weight": AUX_W,
                           "n_fig": N_FIG, "n_motif": N_MOTIF, "n_examples": len(examples)})
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        vocab.save(OUT_DIR / "vocab.json")

        def checkpoint():
            torch.save({"model": model.state_dict(), "config": cfg.__dict__}, OUT_DIR / "model.pt")

        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        start = time.time()
        vloss, best = float("nan"), float("inf")
        for step in range(1, a.steps + 1):
            idx = [rng.randrange(len(train)) for _ in range(a.batch)]
            loss, parts = loss_fn(model, *make_batch(train, idx, vocab, rng, a.enc_block, a.dec_block, a.keep, device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % a.eval_every == 0 or step == 1:
                vloss, vcore = evaluate(model, val, vocab, rng, a.enc_block, a.dec_block, a.keep, device, a.batch)
                mlflow.log_metrics({"train_loss": loss.item(), "val_loss": vloss, "val_core": vcore,
                                    **{f"train_{k}": v.item() for k, v in parts.items()}}, step=step)
                print(f"step {step:5d}  train {loss.item():.3f}  val {vloss:.3f}  core {vcore:.3f}  "
                      f"[{' '.join(f'{k}{v.item():.2f}' for k, v in parts.items())}]  "
                      f"({(time.time()-start)/step*1000:.0f} ms/step)", flush=True)
                if vcore < best:                      # select the best checkpoint by CORE val (melody quality)
                    best = vcore; checkpoint()
        checkpoint()
        mlflow.log_metric("val_loss_final", vloss)
        print(f"saved checkpoint to {OUT_DIR}  (best core-val {best:.3f})", flush=True)


if __name__ == "__main__":
    main()
