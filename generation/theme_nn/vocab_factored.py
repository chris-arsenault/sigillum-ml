"""Factored tokenizer (v1): a position is a BUNDLE of small fields, not one big word.

Each sequence position carries (kind, degree, alteration, octave, duration, cond). The model
embeds each field separately and sums them, and predicts each field with its own head — so it
learns "degree 5" and "a quarter note" ONCE, from every note that uses them, instead of learning
4,170 independent degree×octave×duration combo-words. Far fewer parameters, far more data-efficient.

  kinds: BOS, KEY, SHAPE, CHAR (conditioning) | EVENT (a note/rest) | EOS
  EVENT carries degree(0=rest,1-7) + alteration + octave + duration; conditioning carries `cond`.
"""
from __future__ import annotations

import json
from pathlib import Path

KINDS = ["BOS", "KEY", "SHAPE", "CHAR", "EVENT", "EOS"]
KIND = {k: i for i, k in enumerate(KINDS)}
DEGREE_SIZE = 8   # 0 = rest, 1-7 = scale degree
ALT_SIZE = 3      # 0 diatonic, 1-2 chromatic
FIELDS = ["kind", "degree", "alt", "oct", "dur", "cond"]

# A fixed musical duration vocabulary (in beats), keeping 16ths and triplets, capped at 4 beats.
# Durations are binned to the nearest value — a small field instead of the 218 raw quantized values,
# which is the same data-efficiency win as factoring the rest of the note.
DUR_PALETTE = [1 / 12, 1 / 6, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]


def _dur_idx(duration: float) -> int:
    return min(range(len(DUR_PALETTE)), key=lambda i: abs(DUR_PALETTE[i] - float(duration)))


class FactoredVocab:
    def __init__(self, oct_map: dict, cond_map: dict):
        self.oct = oct_map                      # octave value -> idx
        self.cond = cond_map                    # "key:..|shape:..|char:.." -> idx (0 = <none>)
        self.oct_inv = {i: v for v, i in oct_map.items()}
        self.dur_inv = {i: v for i, v in enumerate(DUR_PALETTE)}

    @classmethod
    def from_examples(cls, examples) -> "FactoredVocab":
        octs, conds = set(), set()
        for ex in examples:
            conds.add(f"key:{ex['key'][0]}:{ex['key'][1]}")
            conds.add(f"shape:{ex['shape']}")
            conds.update(f"char:{c}" for c in ex.get("character", []))
            for _d, _a, octave, _duration in ex["events"]:
                octs.add(int(octave))
        oct_map = {v: i for i, v in enumerate(sorted(octs))}
        cond_map = {"<none>": 0}
        for v in sorted(conds):
            cond_map[v] = len(cond_map)
        return cls(oct_map, cond_map)

    def field_sizes(self) -> dict[str, int]:
        return {"kind": len(KINDS), "degree": DEGREE_SIZE, "alt": ALT_SIZE,
                "oct": len(self.oct), "dur": len(DUR_PALETTE), "cond": len(self.cond)}

    def encode_example(self, ex: dict) -> tuple[list[tuple], int]:
        pos: list[tuple] = [(KIND["BOS"], 0, 0, 0, 0, 0)]
        pos.append((KIND["KEY"], 0, 0, 0, 0, self.cond[f"key:{ex['key'][0]}:{ex['key'][1]}"]))
        pos.append((KIND["SHAPE"], 0, 0, 0, 0, self.cond[f"shape:{ex['shape']}"]))
        for c in ex.get("character", []):
            cidx = self.cond.get(f"char:{c}")
            if cidx is not None:
                pos.append((KIND["CHAR"], 0, 0, 0, 0, cidx))
        prefix_len = len(pos)
        for degree, alt, octave, duration in ex["events"]:
            oi = self.oct.get(int(octave))
            if oi is None:
                continue
            pos.append((KIND["EVENT"], int(degree), int(alt), oi, _dur_idx(duration), 0))
        pos.append((KIND["EOS"], 0, 0, 0, 0, 0))
        return pos, prefix_len

    def cond_id(self, kind: str, value) -> int | None:
        return self.cond.get(f"{kind}:{value}")

    def save(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "oct": {str(k): v for k, v in self.oct.items()},
            "cond": self.cond,
        }))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "FactoredVocab":
        d = json.loads(Path(path).read_text())
        return cls({int(k): v for k, v in d["oct"].items()}, d["cond"])
