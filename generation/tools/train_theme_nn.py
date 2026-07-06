"""Train the baseline ThemeGPT on the encoded dataset.

    python -m generation.tools.build_theme_dataset                  # make outputs/datasets/theme_nn.jsonl first
    python -m generation.tools.train_theme_nn --steps 3000

Loss is taken only on the event region (the conditioning prefix is given, not predicted). Saves a
checkpoint (model + config + vocab) to outputs/models/theme_nn/.
"""
import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from generation.theme_nn.dataset import DEFAULT_OUT
from generation.theme_nn.model import GPTConfig, ThemeGPT
from generation.theme_nn.vocab import Vocab, load_examples

OUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "models" / "theme_nn"


def encode_all(examples, vocab, block_size):
    data = []
    for ex in examples:
        ids, prefix = vocab.encode_example(ex)
        ids = ids[:block_size]
        if len(ids) > prefix + 1:          # need at least one predicted event
            data.append((ids, prefix))
    return data


def make_batch(data, indices, block_size, pad, device):
    x = torch.full((len(indices), block_size), pad, dtype=torch.long)
    mask = torch.zeros((len(indices), block_size - 1), dtype=torch.bool)
    for b, i in enumerate(indices):
        ids, prefix = data[i]
        x[b, : len(ids)] = torch.tensor(ids)
        mask[b, prefix - 1 : len(ids) - 1] = True   # predict events + eos only
    return x.to(device), mask.to(device)


@torch.no_grad()
def evaluate(model, data, block_size, pad, device, batch, rounds=10):
    model.eval()
    total, count = 0.0, 0
    for _ in range(rounds):
        idx = [random.randrange(len(data)) for _ in range(min(batch, len(data)))]
        x, mask = make_batch(data, idx, block_size, pad, device)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits[mask], x[:, 1:][mask])
        total += loss.item(); count += 1
    model.train()
    return total / max(count, 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_OUT))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = "cpu"
    examples = load_examples(Path(args.data))
    random.shuffle(examples)
    split = max(1, int(len(examples) * 0.95))
    train_ex, val_ex = examples[:split], examples[split:] or examples[-4:]
    vocab = Vocab.from_examples(examples)
    train = encode_all(train_ex, vocab, args.block_size)
    val = encode_all(val_ex, vocab, args.block_size)
    print(f"examples: {len(examples)} ({len(train)} train / {len(val)} val)   vocab: {len(vocab)}")

    config = GPTConfig(vocab_size=len(vocab), block_size=args.block_size,
                       n_layer=args.layers, n_head=args.heads, d_model=args.d_model)
    model = ThemeGPT(config).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    start = time.time()
    for step in range(1, args.steps + 1):
        idx = [random.randrange(len(train)) for _ in range(args.batch)]
        x, mask = make_batch(train, idx, args.block_size, vocab.pad, device)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits[mask], x[:, 1:][mask])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == 1:
            vloss = evaluate(model, val, args.block_size, vocab.pad, device, args.batch)
            print(f"step {step:5d}  train {loss.item():.3f}  val {vloss:.3f}  "
                  f"({(time.time() - start) / step * 1000:.0f} ms/step)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config.__dict__}, OUT_DIR / "model.pt")
    vocab.save(OUT_DIR / "vocab.json")
    print(f"saved checkpoint to {OUT_DIR}")


if __name__ == "__main__":
    main()
