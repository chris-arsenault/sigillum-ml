"""Decode: given a kernel (anchors + free spans + control), fill the spans and land the pins.

DIATONIC-INTERVAL-PRIMARY generation: the decoder predicts the next move as a DIATONIC STEP (along
the scale) plus an ALTERATION (chromatic accidental), not an absolute pitch. Each gap's pitch is the
running scale position from the left pin — so every note lands on a scale tone (no chromatic drift),
chromaticism comes only from the per-note alteration, and the last note is clamped to land the next
pin (a gap is a traversal, so the line can't sit on one note). A diatonic step of "none" is a rest.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from framework.analysis.tonal import SCALE_OFFSETS
from generation.theme_nn.infill import ALT_SIZE, BAR_SIZE, BEAT_SIZE, IKIND, InfillVocab, _gap_bin
from generation.theme_nn.model_infill import FactoredEncDec, InfillConfig
from generation.theme_nn.representation import Event, decode
from generation.theme_nn.vocab_factored import DUR_PALETTE, _dur_idx

_GRID = 1 / 12
_REST_STEP = 29        # the diatonic-step head's "none" class = a rest (no pitch move)
_DPOS_LO, _DPOS_HI = 21, 62   # clamp the scale position to octaves ~3-8 (dpos = octave*7 + degree-1)


def _slot(onset: float, barlen: float) -> tuple[int, int]:
    slots = max(1, int(round(barlen / _GRID)))
    return min(int(round((onset % barlen) / _GRID)) % slots, BEAT_SIZE - 1), min(int(onset // barlen), BAR_SIZE - 1)


def load_model(checkpoint_dir: Path):
    ck = torch.load(Path(checkpoint_dir) / "model.pt", map_location="cpu", weights_only=False)
    model = FactoredEncDec(InfillConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.eval()
    return model, InfillVocab.load(Path(checkpoint_dir) / "vocab.json")


def _ctrl(kind, cond=0):
    return (IKIND[kind], 0, 0, 0, 0, cond, 0, 0)


def _event_pos(vocab, deg, alt, octave, dur, beat=0, bar=0):
    return (IKIND["EVENT"], int(deg), int(alt), vocab.oct.get(int(octave), 0), _dur_idx(dur), 0,
            min(int(beat), BEAT_SIZE - 1), min(int(bar), BAR_SIZE - 1))


def build_encoder(vocab, spec, barlen) -> list[tuple]:
    enc = [_ctrl("BOS"),
           _ctrl("KEY", vocab.cond.get(f"key:{spec['key'][0]}:{spec['key'][1]}", 0)),
           _ctrl("SHAPE", vocab.cond.get(f"shape:{spec['shape']}", 0))]
    for c in spec.get("character", []):
        enc.append(_ctrl("CHAR", vocab.cond.get(f"char:{c}", 0)))
    for r in spec.get("harmony", []):
        enc.append(_ctrl("HARM", vocab.cond.get(f"harm:{r}", 0)))
    onset = 0.0
    for kind, payload in spec["segments"]:
        if kind == "anchor":
            for deg, alt, octave, dur in payload:
                beat, bar = _slot(onset, barlen)
                enc.append(_event_pos(vocab, deg, alt, octave, dur, beat, bar))
                onset += float(dur)
        else:
            enc.append(_ctrl("GAP", vocab.cond.get(f"gap:{_gap_bin(payload)}", 0)))
            onset += float(payload)
    return enc


def fill(model, vocab, spec, *, temperature=0.9, seed=0, min_dur=1 / 12) -> list[tuple]:
    """Return the full melody as (degree, alt, octave, duration) events (anchors + tiled fills)."""
    torch.manual_seed(seed)
    barlen = float(spec.get("barlen", 4.0))
    tonic, mode = spec["key"]
    segs = spec["segments"]

    def dpos_of(deg, octave):
        return octave * 7 + (deg - 1)

    memory = model.encode(torch.tensor([build_encoder(vocab, spec, barlen)], dtype=torch.long))
    dec = [_ctrl("BOS")]

    def sample(logits):
        return int(torch.multinomial(F.softmax(logits / max(temperature, 1e-6), dim=-1), 1))

    fills, onset, prev_dpos = [], 0.0, None
    for si, (kind, payload) in enumerate(segs):
        if kind == "anchor":
            for deg, alt, octave, dur in payload:
                if deg:
                    prev_dpos = dpos_of(deg, octave)
                onset += float(dur)
            continue
        target = float(payload)
        land = None                                   # (scale-position, alteration) of the next anchor
        for nk, np_ in segs[si + 1:]:
            if nk == "anchor" and np_ and np_[0][0]:
                d0, a0, o0, _ = np_[0]
                land = (dpos_of(d0, o0), a0); break
        dec.append(_ctrl("GAP", vocab.cond.get(f"gap:{_gap_bin(target)}", 0)))
        acc, span = 0.0, []
        while acc < target - 1e-3 and len(span) < 256:
            out = model.decode(torch.tensor([dec], dtype=torch.long), memory)
            step = sample(out["dstep"][0, -1])
            alt = min(sample(out["alt"][0, -1]), ALT_SIZE - 1)
            duri = sample(out["dur"][0, -1])
            dur = DUR_PALETTE[duri]
            last = acc + dur > target - 1e-9
            if last:
                dur = round(target - acc, 4)
            if dur < min_dur / 2:
                break
            beat, bar = _slot(onset + acc, barlen)
            if step == _REST_STEP:                    # a rest: no pitch move
                deg, alt, octi, octave = 0, 0, 0, 0
            else:
                if last and land is not None:         # land the pin: traverse to the next anchor
                    next_dpos, alt = land
                else:
                    base = prev_dpos if prev_dpos is not None else dpos_of(1, 5)
                    next_dpos = base + (step - 14)
                next_dpos = max(_DPOS_LO, min(_DPOS_HI, next_dpos))
                deg, octave = next_dpos % 7 + 1, next_dpos // 7
                octi = vocab.oct.get(octave, 0)
                prev_dpos = next_dpos
            span.append((deg, alt, octave, dur)); acc += dur
            dec.append((IKIND["EVENT"], deg, alt, octi, duri, 0, min(beat, BEAT_SIZE - 1),
                        min(bar, BAR_SIZE - 1)))
        fills.append(span)
        onset += target

    full, gi = [], 0
    for kind, payload in segs:
        if kind == "anchor":
            full += list(payload)
        else:
            full += fills[gi]; gi += 1
    return full


def to_melody(events, key) -> list[tuple]:
    """(degree, alt, octave, duration) events -> (midi|None, duration) via the key-relative decode."""
    evs = [Event("rest", None, 0, 0, d) if deg == 0 else Event("note", deg, a, o, d)
           for deg, a, o, d in events]
    return decode(evs, key)
