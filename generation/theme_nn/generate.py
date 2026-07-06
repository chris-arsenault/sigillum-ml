"""Generation + decode for the trained ThemeGPT: conditioning -> token ids -> Events -> melody.

The inverse of the training encode path. Builds a conditioning prefix (key + shape + character),
samples event tokens from the model, and decodes them back to a melody item-list. No pins /
infilling / bar-tiling yet — this is the free baseline, so we can finally listen.
"""
from __future__ import annotations

from pathlib import Path

import torch

from generation.theme_nn.model import GPTConfig, ThemeGPT
from generation.theme_nn.representation import Event, MelodyItem, decode
from generation.theme_nn.vocab import Vocab


def load_model(checkpoint_dir: Path) -> tuple[ThemeGPT, Vocab]:
    ckpt = torch.load(Path(checkpoint_dir) / "model.pt", map_location="cpu", weights_only=False)
    model = ThemeGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, Vocab.load(Path(checkpoint_dir) / "vocab.json")


def _prefix_ids(vocab: Vocab, key: tuple[int, str], shape: str, character: list[str]) -> list[int]:
    tokens = ["<bos>", f"key:{key[0]}:{key[1]}", f"shape:{shape}"] + [f"char:{c}" for c in character]
    missing = [t for t in tokens if t not in vocab.stoi]
    if missing:
        print(f"  note: conditioning tokens not seen in training, ignored: {missing}")
    return [vocab.stoi[t] for t in tokens if t in vocab.stoi]


def _parse_event(token: str) -> Event | None:
    if not token.startswith("ev:"):
        return None
    _, degree, alteration, octave, duration = token.split(":")
    degree = int(degree)
    if degree == 0:
        return Event("rest", None, 0, 0, float(duration))
    return Event("note", degree, int(alteration), int(octave), float(duration))


def generate_melody(model: ThemeGPT, vocab: Vocab, *, key: tuple[int, str], shape: str,
                    character: list[str], temperature: float = 1.0, top_k: int = 40,
                    max_events: int = 200, seed: int | None = None) -> list[MelodyItem]:
    if seed is not None:
        torch.manual_seed(seed)
    prefix = _prefix_ids(vocab, key, shape, character)
    idx = torch.tensor([prefix], dtype=torch.long)
    out = model.generate(idx, max_events, vocab.eos, temperature=temperature, top_k=top_k)
    events: list[Event] = []
    for token_id in out[0, len(prefix):].tolist():
        if token_id == vocab.eos:
            break
        event = _parse_event(vocab.itos[token_id])
        if event is not None:
            events.append(event)
    return decode(events, key)
