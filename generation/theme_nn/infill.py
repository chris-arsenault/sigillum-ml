"""Infilling task + representation (the designed model): fill the free spans of a kernel.

A kernel is fixed anchors (pinned notes at their positions) + a harmonic/shape/character frame, with
free spans between the anchors. We train a T5-style span-infilling model: encode the kernel (control
+ anchor events + a GAP sentinel per free span, each carrying the span's beat-length), and the
decoder generates each gap's contents in order. Training examples are made by masking spans of real
melodies down to sparse anchors — so the model learns to fill between pins, not to free-run.

  encoder (the kernel):  BOS KEY SHAPE CHAR.. HARM(bar0).. | EVENT(anchor).. GAP(len).. (melodic order)
  decoder (the fills):   BOS  GAP(len0) <span0 events..>  GAP(len1) <span1 events..>  EOS

Reuses the factored field bundle (kind, degree, alt, oct, dur, cond) from vocab_factored; adds the
HARM and GAP kinds and the harmony / gap-length cond values.
"""
from __future__ import annotations

import json
from pathlib import Path
from random import Random

from generation.theme_nn.vocab_factored import DUR_PALETTE, _dur_idx

# Kinds for the infilling sequences (superset of the factored kinds).
INFILL_KINDS = ["BOS", "KEY", "SHAPE", "CHAR", "HARM", "EVENT", "GAP", "EOS"]
IKIND = {k: i for i, k in enumerate(INFILL_KINDS)}
DEGREE_SIZE = 8
ALT_SIZE = 3
# INPUT fields the model conditions on (metric position — context, not something it predicts):
#   beat = metric slot within the bar (1/12 grid: 0=downbeat, distinguishes the offbeat 16th)
#   bar  = bar index from the melody start (clamped)
# Interval + chord-position are NOT inputs; they are étude targets (see below) — predicting them
# pushes the model's own note choices toward melodic/harmonic coherence, which feeding them can't.
BEAT_SIZE = 96    # up to 8-beat bars (8*12); non-4/4 metres land here, 4/4 uses 0..47
BAR_SIZE = 64
IVL_SIZE = 26     # 0..24 = interval -12..+12, 25 = none (first note / after the boundary)
CPOS_CLASSES = ["rest", "R", "3", "5", "7", "nct"]
_CPOS_IDX = {c: i for i, c in enumerate(CPOS_CLASSES)}
CPOS_SIZE = len(CPOS_CLASSES)
FIELDS = ["kind", "degree", "alt", "oct", "dur", "cond", "beat", "bar"]


def ivl_idx(interval: int) -> int:
    return 25 if int(interval) == 99 else max(-12, min(12, int(interval))) + 12


def cpos_idx(label: str) -> int:
    return _CPOS_IDX.get(label, 0)


# Diatonic-step interval: the PRIMARY pitch move, measured in SCALE STEPS (not semitones), so the
# reconstructed pitch lands on a scale tone by construction (no chromatic drift). Chromatic notes
# come from the separate per-note `alt` field. Range +-14 steps (= +-2 octaves), 29 = none/rest.
DSTEP_SIZE = 30


def dstep_idx(step: int) -> int:
    return 29 if int(step) == 99 else max(-14, min(14, int(step))) + 14


def _diatonic_steps(rows) -> list[int]:
    """Per-row diatonic-step interval from the stored degree+octave: diatonic position is
    octave*7 + (degree-1); the step is the change in it between consecutive NOTES (99 = none —
    rests/first; state carries across rests)."""
    out, prev = [], None
    for r in rows:
        deg, octave = r[0][0], int(r[0][2])
        if deg == 0:                                # a rest
            out.append(99); continue
        dpos = octave * 7 + (deg - 1)
        out.append(99 if prev is None else dpos - prev)
        prev = dpos
    return out


def _higher_order(rows) -> tuple[list[int], list[int]]:
    """2nd- and 3rd-order melodic interval per row, derived from the stored 1st-order interval.
    i2[k] = i1[k] - i1[k-1] over consecutive NOTES; i3[k] = i2[k] - i2[k-1]. 99 = undefined (rests,
    first notes, or before the prior order exists); the running state carries across rests."""
    i2s, i3s, prev_i1, prev_i2 = [], [], None, None
    for r in rows:
        deg, i1 = r[0][0], r[5]
        if deg == 0 or i1 == 99:                # a rest or first-note sentinel: no interval
            i2s.append(99); i3s.append(99)
            continue
        i2 = 99 if prev_i1 is None else i1 - prev_i1
        i3 = 99 if (i2 == 99 or prev_i2 is None) else i2 - prev_i2
        i2s.append(i2); i3s.append(i3)
        prev_i1 = i1
        if i2 != 99:
            prev_i2 = i2
    return i2s, i3s

# Gap length (free-span duration) in beats, binned — a GAP sentinel carries the span it must fill.
GAP_PALETTE = [0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]

# Auxiliary (étude) targets the decoder PREDICTS for each fill note — figure, motif transform,
# melodic interval, and chord-position. NEVER model inputs: predicting them from the trunk's hidden
# state presses figurative / recurrence / melodic-step / harmonic-function structure into the
# weights, which shapes the note choices the model generates (feeding them could not). Closed label
# sets; the extractors' raw values map here (motif's "transpose±N" all collapse to "transpose").
FIG_CLASSES = ["rest", "step", "run-up", "run-dn", "arp-up", "arp-dn", "trill",
               "neighbor-up", "neighbor-dn", "leap", "repeat", "sustained"]
MOTIF_CLASSES = ["-", "exact", "transpose", "inversion", "tonal-transpose", "tonal-inversion",
                 "retrograde", "augmentation", "diminution"]
_FIG_IDX = {c: i for i, c in enumerate(FIG_CLASSES)}
_MOTIF_IDX = {c: i for i, c in enumerate(MOTIF_CLASSES)}
N_FIG, N_MOTIF = len(FIG_CLASSES), len(MOTIF_CLASSES)


def fig_idx(label: str) -> int:
    return _FIG_IDX.get(label, _FIG_IDX["step"])


def motif_idx(label: str) -> int:
    return _MOTIF_IDX["transpose"] if str(label).startswith("transpose") else _MOTIF_IDX.get(label, 0)


def _gap_bin(beats: float) -> int:
    return min(range(len(GAP_PALETTE)), key=lambda i: abs(GAP_PALETTE[i] - beats))


def _spans(n_events: int, rng: Random, keep_fraction: float):
    """Partition event indices [0,n) into alternating keep(anchor)/mask(gap) runs, mimicking a
    kernel's sparse pins. Returns a list of (is_mask, start, end) runs covering the melody."""
    runs, i = [], 0
    # always open with a kept anchor (a head pin), then alternate, biased to the keep_fraction
    keep = True
    while i < n_events:
        remaining = n_events - i
        target = max(1, int(round((2 if keep else 3) * (keep_fraction if keep else 1 - keep_fraction) * 2)))
        length = max(1, min(remaining, rng.randint(1, max(2, target))))
        runs.append((not keep, i, i + length))
        i += length
        keep = not keep
    if not any(m for m, _, _ in runs):           # ensure at least one gap to fill
        runs[-1] = (True, runs[-1][1], runs[-1][2])
    return runs


class InfillVocab:
    """Field vocabularies for the infilling sequences. `cond` covers key/shape/char/harmony/gap."""

    def __init__(self, oct_map: dict, cond_map: dict):
        self.oct = oct_map
        self.cond = cond_map
        self.oct_inv = {i: v for v, i in oct_map.items()}
        self.dur_inv = {i: v for i, v in enumerate(DUR_PALETTE)}

    @classmethod
    def from_examples(cls, examples) -> "InfillVocab":
        octs, conds = set(), set()
        for ex in examples:
            conds.add(f"key:{ex['key'][0]}:{ex['key'][1]}")
            conds.add(f"shape:{ex['shape']}")
            conds.update(f"char:{c}" for c in ex.get("character", []))
            conds.update(f"harm:{r}" for r in ex.get("harmony", []))
            conds.update(f"gap:{i}" for i in range(len(GAP_PALETTE)))
            for _d, _a, octave, _du in ex["events"]:
                octs.add(int(octave))
        oct_map = {v: i for i, v in enumerate(sorted(octs))}
        cond_map = {"<none>": 0}
        for v in sorted(conds):
            cond_map[v] = len(cond_map)
        return cls(oct_map, cond_map)

    def field_sizes(self) -> dict:
        return {"kind": len(INFILL_KINDS), "degree": DEGREE_SIZE, "alt": ALT_SIZE,
                "oct": len(self.oct), "dur": len(DUR_PALETTE), "cond": len(self.cond),
                "beat": BEAT_SIZE, "bar": BAR_SIZE}

    def _event(self, degree, alt, octave, duration, beat=0, bar=0) -> tuple:
        return (IKIND["EVENT"], int(degree), int(alt), self.oct.get(int(octave), 0),
                _dur_idx(duration), 0, min(int(beat), BEAT_SIZE - 1), min(int(bar), BAR_SIZE - 1))

    # Non-EVENT tokens (BOS/KEY/SHAPE/CHAR/HARM/GAP/EOS) carry the "none" metric slots.
    def _ctrl(self, kind, cond=0) -> tuple:
        return (IKIND[kind], 0, 0, 0, 0, cond, 0, 0)

    def _cond(self, kind, value) -> tuple:
        return self._ctrl(kind, self.cond.get(f"{kind.lower()}:{value}", 0))

    def encode_pair(self, ex: dict, rng: Random, keep_fraction: float = 0.4):
        """Build (encoder kernel positions, decoder fill positions, decoder aux targets) for one
        example. ``aux`` is one (fig, motif, cpos, dstep) tuple per decoder position — set only on
        EVENT fills, 0 elsewhere. ``dstep`` (the diatonic-step interval) is the primary pitch target
        the decoder predicts; fig/motif/cpos are étude targets that shape the trunk."""
        fig, mot = ex.get("figuration") or [], ex.get("motif") or []
        beats, bars = ex.get("beat") or [], ex.get("bar") or []
        cpos = ex.get("chordpos") or []
        at = lambda seq, i, default: seq[i] if i < len(seq) else default
        rows = [(e, at(fig, i, "step"), at(mot, i, "-"), at(beats, i, 0), at(bars, i, 0),
                 0, at(cpos, i, "rest"))
                for i, e in enumerate(ex["events"]) if self.oct.get(int(e[2])) is not None]
        if len(rows) < 6:
            return None
        dsteps = _diatonic_steps(rows)     # primary diatonic-step interval per row (99 = none)
        events = [r[0] for r in rows]
        runs = _spans(len(events), rng, keep_fraction)
        ev = lambda k: self._event(*events[k], rows[k][3], rows[k][4])   # event with its metric slot
        none_aux = (0, 0, cpos_idx("rest"), dstep_idx(99))

        enc = [self._ctrl("BOS"),
               self._cond("KEY", f"{ex['key'][0]}:{ex['key'][1]}"),
               self._cond("SHAPE", ex["shape"])]
        for c in ex.get("character", []):
            enc.append(self._cond("CHAR", c))
        for r in ex.get("harmony", []):
            enc.append(self._cond("HARM", r))

        dec = [self._ctrl("BOS")]
        aux = [none_aux]
        for is_mask, s, e in runs:
            if is_mask:
                gap_beats = sum(float(events[k][3]) for k in range(s, e))
                gid = self.cond.get(f"gap:{_gap_bin(gap_beats)}", 0)
                enc.append(self._ctrl("GAP", gid))      # a free span to fill
                dec.append(self._ctrl("GAP", gid)); aux.append(none_aux)
                for k in range(s, e):
                    dec.append(ev(k))
                    aux.append((fig_idx(rows[k][1]), motif_idx(rows[k][2]),
                                cpos_idx(rows[k][6]), dstep_idx(dsteps[k])))
            else:
                for k in range(s, e):
                    enc.append(ev(k))                                  # an anchor (pin)
        dec.append(self._ctrl("EOS")); aux.append(none_aux)
        return enc, dec, aux

    def save(self, path: Path) -> Path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"oct": {str(k): v for k, v in self.oct.items()},
                                          "cond": self.cond}))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "InfillVocab":
        d = json.loads(Path(path).read_text())
        return cls({int(k): v for k, v in d["oct"].items()}, d["cond"])
