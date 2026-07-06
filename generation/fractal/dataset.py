"""Training windows for the detailer: 16-beat melody windows from the curated corpus, KEY-NORMALIZED
to C and carrying everything the v2 model conditions on.

Each window yields: notes (midi|None, dur, velocity) transposed so the key's tonic = C; per-note
figuration labels (for ornament-strip coarsening); the window's character tags + per-bar harmony +
note-density (for conditioning). Read from the theme_nn jsonl (events/velocity/figuration/key/
character/harmony already extracted) and decoded — no re-analysis.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from generation.fractal.roll import BEATS
from generation.theme_nn.dataset import DEFAULT_OUT
from generation.theme_nn.representation import Event, decode

# Conditioning vocabularies. Character = the categorisation taxonomy; harmony = Roman numerals.
CHAR_TAGS = ["battle", "boss", "overworld", "town", "dungeon", "character", "romance", "sad",
             "victory", "tension", "title", "ending", "ambient", "comedy", "heroic", "march",
             "dance", "chorale", "fugue", "virtuosic", "aria", "nocturne"]
CHAR_IDX = {c: i + 1 for i, c in enumerate(CHAR_TAGS)}        # 0 = pad/none
HARM_RN = ["I", "i", "II", "ii", "III", "iii", "IV", "iv", "V", "v", "VI", "vi", "VII", "vii",
           "ii°", "vii°", "N", "Ger", "It", "Fr", "?"]
HARM_IDX = {r: i + 1 for i, r in enumerate(HARM_RN)}          # 0 = pad/none
N_CHAR, N_HARM = len(CHAR_TAGS) + 1, len(HARM_RN) + 1
_MIN_NOTES = 12
_BPB = 4


@dataclass
class Window:
    notes: list[tuple]            # (midi|None, dur, velocity), key-normalised (tonic -> C)
    figs: list[str]               # per-note figuration label (aligned to notes)
    character: list[int]          # character tag ids
    harmony: list[int]            # per-bar Roman-numeral ids (this window's bars)
    density: float                # sounding notes per beat


def _decoded(ex):
    """Decode one example's events into key-normalised (midi|None, dur, vel) + per-note figuration."""
    tonic, mode = ex["key"]
    vels, figl = ex.get("velocity") or [], ex.get("figuration") or []
    items, figs = [], []
    for i, (dg, a, o, d) in enumerate(ex["events"]):
        if dg == 0:
            items.append((None, float(d), 0)); figs.append("rest")
        else:
            m = decode([Event("note", dg, a, o, d)], (tonic, mode))[0][0]
            items.append((m - tonic, float(d), int(vels[i]) if i < len(vels) else 64))  # tonic -> C
            figs.append(figl[i] if i < len(figl) else "step")
    return items, figs


def melody_windows(limit: int | None = None, path=DEFAULT_OUT) -> list[Window]:
    out: list[Window] = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            items, figs = _decoded(ex)
            char = [CHAR_IDX[c] for c in ex.get("character", []) if c in CHAR_IDX]
            harm = ex.get("harmony", [])
            beat = 0.0
            cur, curf = [], []
            wstart_bar = 0
            for (note, fig) in zip(items, figs):
                cur.append(note); curf.append(fig); beat += note[1]
                if beat >= BEATS - 1e-6:
                    bars = harm[wstart_bar:wstart_bar + BEATS // _BPB]
                    hids = [HARM_IDX.get(r, HARM_IDX["?"]) for r in bars]
                    nnotes = sum(1 for m, _d, _v in cur if m is not None)
                    if nnotes >= _MIN_NOTES:
                        out.append(Window(cur, curf, char, hids, nnotes / BEATS))
                    cur, curf, beat = [], [], 0.0
                    wstart_bar += BEATS // _BPB
            if limit and len(out) >= limit:
                return out[:limit]
    return out
