"""Tokenizer: a training example <-> a flat sequence of integer token ids.

A sequence is a conditioning prefix the model is TOLD, then the melody EVENTS it generates:

    <bos> key:<tonic>:<mode> shape:<archetype> char:<tag>...  ev:<deg>:<alt>:<oct>:<dur> ... <eos>

v1 conditions on key + shape + character; per-bar harmony conditioning is the next phase (the
examples already carry it). One token per event keeps the first model simple; loss is taken only
on the event region (the prefix is given, not predicted).
"""
from __future__ import annotations

import json
from pathlib import Path

SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _prefix_tokens(example: dict) -> list[str]:
    key = example["key"]
    tokens = ["<bos>", f"key:{key[0]}:{key[1]}", f"shape:{example['shape']}"]
    tokens += [f"char:{c}" for c in example.get("character", [])]
    return tokens


def _event_token(event) -> str:
    return f"ev:{event[0]}:{event[1]}:{event[2]}:{event[3]}"


class Vocab:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: s for s, i in stoi.items()}

    @property
    def pad(self) -> int: return self.stoi["<pad>"]
    @property
    def bos(self) -> int: return self.stoi["<bos>"]
    @property
    def eos(self) -> int: return self.stoi["<eos>"]

    def __len__(self) -> int:
        return len(self.stoi)

    @classmethod
    def from_examples(cls, examples) -> "Vocab":
        tokens = list(SPECIALS)
        seen = set(tokens)
        for example in examples:
            for token in _prefix_tokens(example) + [_event_token(e) for e in example["events"]]:
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        return cls({token: i for i, token in enumerate(tokens)})

    def encode_example(self, example: dict) -> tuple[list[int], int]:
        """(token ids for the full sequence, length of the conditioning prefix)."""
        prefix = _prefix_tokens(example)
        events = [_event_token(e) for e in example["events"]]
        unk = self.stoi["<unk>"]
        ids = [self.stoi.get(t, unk) for t in prefix + events + ["<eos>"]]
        return ids, len(prefix)

    def save(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stoi))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(Path(path).read_text()))


def load_examples(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
