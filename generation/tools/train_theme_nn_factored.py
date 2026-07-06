"""Train the factored ThemeGPT (v1) with MLflow tracking.

    python -m generation.tools.train_theme_nn_factored --steps 1500

Per-event loss = sum of the field cross-entropies (kind + degree + alt + oct + dur), comparable to
v0's single-token CE (~3.9). Logs params/metrics to MLflow (mlruns/); saves to outputs/models/theme_nn_v1/.
"""
import argparse
import random
import time
from pathlib import Path

import mlflow
import torch
import torch.nn.functional as F

from generation.theme_nn.dataset import DEFAULT_OUT
from generation.theme_nn.model_factored import PRED, FactoredConfig, FactoredThemeGPT
from generation.theme_nn.vocab_factored import FIELDS, KIND
from generation.theme_nn.vocab import load_examples

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models" / "theme_nn_v1"
FI = {f: i for i, f in enumerate(FIELDS)}
NF = len(FIELDS)


def encode_all(examples, vocab, block_size):
    data = []
    for ex in examples:
        pos, prefix = vocab.encode_example(ex)
        pos = pos[:block_size]
        if len(pos) > prefix + 1:
            data.append((pos, prefix))
    return data


def make_batch(data, indices, block_size, device):
    X = torch.zeros((len(indices), block_size, NF), dtype=torch.long)
    base = torch.zeros((len(indices), block_size - 1), dtype=torch.bool)   # predict events+eos
    for b, i in enumerate(indices):
        pos, prefix = data[i]
        for t, p in enumerate(pos):
            X[b, t] = torch.tensor(p)
        base[b, prefix - 1 : len(pos) - 1] = True
    return X.to(device), base.to(device)


def loss_fn(model, X, base):
    logits = model(X[:, :-1])
    targets = X[:, 1:]                                  # (B, T-1, NF)
    tgt_kind = targets[:, :, FI["kind"]]
    event = base & (tgt_kind == KIND["EVENT"])          # event-only fields
    losses = {}
    losses["kind"] = F.cross_entropy(logits["kind"][base], tgt_kind[base])
    for f in ("degree", "alt", "oct", "dur"):
        losses[f] = F.cross_entropy(logits[f][event], targets[:, :, FI[f]][event])
    return sum(losses.values()), losses


@torch.no_grad()
def evaluate(model, data, block_size, device, batch, rounds=10):
    model.eval()
    total = 0.0
    for _ in range(rounds):
        idx = [random.randrange(len(data)) for _ in range(min(batch, len(data)))]
        X, base = make_batch(data, idx, block_size, device)
        total += loss_fn(model, X, base)[0].item()
    model.train()
    return total / rounds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_OUT))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--block-size", type=int, default=768)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from generation.theme_nn.vocab_factored import FactoredVocab
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cpu"
    examples = load_examples(Path(args.data)); random.shuffle(examples)
    split = max(1, int(len(examples) * 0.95))
    vocab = FactoredVocab.from_examples(examples)
    train = encode_all(examples[:split], vocab, args.block_size)
    val = encode_all(examples[split:] or examples[-4:], vocab, args.block_size)
    config = FactoredConfig(field_sizes=vocab.field_sizes(), block_size=args.block_size,
                            n_layer=args.layers, n_head=args.heads, d_model=args.d_model)
    model = FactoredThemeGPT(config).to(device)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"examples: {len(examples)} ({len(train)}/{len(val)})  field_sizes: {vocab.field_sizes()}  params: {params:.1f}M")

    mlflow.set_experiment("theme_nn")
    with mlflow.start_run(run_name="v1_factored"):
        mlflow.log_params({"variant": "factored", "d_model": args.d_model, "layers": args.layers,
                           "heads": args.heads, "block_size": args.block_size, "batch": args.batch,
                           "lr": args.lr, "steps": args.steps, "params_M": round(params, 2),
                           "vocab_total": sum(vocab.field_sizes().values()), "n_examples": len(examples)})
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        start = time.time()
        for step in range(1, args.steps + 1):
            idx = [random.randrange(len(train)) for _ in range(args.batch)]
            X, base = make_batch(train, idx, args.block_size, device)
            loss, parts = loss_fn(model, X, base)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if step % args.eval_every == 0 or step == 1:
                vloss = evaluate(model, val, args.block_size, device, args.batch)
                mlflow.log_metrics({"train_loss": loss.item(), "val_loss": vloss,
                                    **{f"train_{k}": v.item() for k, v in parts.items()}}, step=step)
                print(f"step {step:5d}  train {loss.item():.3f}  val {vloss:.3f}  "
                      f"[{' '.join(f'{k}{v.item():.2f}' for k, v in parts.items())}]  "
                      f"({(time.time()-start)/step*1000:.0f} ms/step)")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": {**config.__dict__}}, OUT_DIR / "model.pt")
        vocab.save(OUT_DIR / "vocab.json")
        mlflow.log_metric("val_loss_final", vloss)
        print(f"saved checkpoint to {OUT_DIR}")


if __name__ == "__main__":
    main()
