"""Piano-roll representation for the monophonic detailer — the "image" the diffusion model works on.

A monophonic melody is a list of ``(midi | None, duration [, velocity])`` items, rendered onto a
time-major 1/12-beat grid (16ths AND triplets land exactly) as channels:

  * P pitch channels (one-hot: which pitch sounds in this cell)
  * rest channel (no pitch sounds)
  * onset channel (first cell of each note — held note vs two repeats)
  * velocity channel (CONTINUOUS loudness, velocity/127, held across the note — the dynamic heatmap)

So a window is a ``(C, T)`` float image, C = P + 3. Binary channels are {0,1} (relaxed to [-1,1] by
the diffusion code); velocity is already continuous. ``strip_ornaments`` makes the coarse input by
removing figuration ornaments (run/trill/neighbour) and absorbing their time into the structural note.
"""
from __future__ import annotations

import numpy as np

GRID = 1 / 12               # cells per beat
PMIN, PMAX = 48, 84         # MIDI range [C3, C6)
P = PMAX - PMIN             # 36 pitch channels
REST, ONSET, VEL = P, P + 1, P + 2
C = P + 3                   # total channels
BEATS = 16
T = int(round(BEATS / GRID))  # 192 cells
FLAT_VEL = 80               # the constant velocity a coarse line carries (model adds the shaping)
_ORNAMENT = {"run-up", "run-dn", "trill", "neighbor-up", "neighbor-dn"}


def _clamp_pitch(midi: int) -> int:
    while midi >= PMAX:
        midi -= 12
    while midi < PMIN:
        midi += 12
    return max(PMIN, min(PMAX - 1, midi))


def melody_to_roll(notes, t_cells: int = T, flat_vel: int | None = None) -> np.ndarray:
    """(midi|None, dur[, vel]) items -> a (C, t_cells) roll. ``flat_vel`` overrides every note's
    velocity with a constant (used to build the flat coarse input the model shapes)."""
    roll = np.zeros((C, t_cells), dtype=np.float32)
    cell = 0
    for item in notes:
        midi, dur = item[0], item[1]
        vel = flat_vel if flat_vel is not None else (item[2] if len(item) > 2 else 64)
        n = int(round(float(dur) / GRID))
        if n <= 0:
            continue
        if cell >= t_cells:
            break
        span = list(range(cell, min(cell + n, t_cells)))
        if midi is None:
            roll[REST, span] = 1.0
        else:
            roll[_clamp_pitch(int(midi)) - PMIN, span] = 1.0
            roll[ONSET, cell] = 1.0
            roll[VEL, span] = max(1, int(vel)) / 127.0
        cell += n
    roll[REST, (roll[:P].sum(0) + roll[REST]) == 0] = 1.0     # unfilled tail = rest
    return roll


def roll_to_melody(roll: np.ndarray) -> list[tuple]:
    """Inverse: (C,T) roll -> (midi|None, duration, velocity) items. Onset-authoritative segmentation;
    pitch = modal pitch over the note; velocity = mean of the velocity channel over the note."""
    from collections import Counter

    t_cells = roll.shape[1]
    pitch_idx = roll[:P].argmax(0)
    is_pitch = roll[:P].max(0) > 0.5
    is_onset = roll[ONSET] > 0.5
    items: list[tuple] = []
    seg, seg_v, seg_pitched = [], [], False

    def flush():
        if not seg:
            return
        if seg_pitched:
            midi = Counter([p for p in seg if p is not None]).most_common(1)[0][0] + PMIN
            vel = int(round(np.clip(np.mean(seg_v), 0, 1) * 127)) or 1
            items.append((midi, len(seg) * GRID, vel))
        else:
            items.append((None, len(seg) * GRID, 0))

    for t in range(t_cells):
        here = int(pitch_idx[t]) if is_pitch[t] else None
        boundary = (is_pitch[t] != seg_pitched) or (is_pitch[t] and is_onset[t])
        if boundary and seg:
            flush(); seg, seg_v = [], []
        seg_pitched = is_pitch[t]
        seg.append(here); seg_v.append(float(roll[VEL, t]))
    flush()
    return items


def strip_ornaments(notes, figs) -> list[tuple]:
    """Coarse input = the structural line: drop figuration ornaments (run/trill/neighbour) and absorb
    their duration into the preceding structural note. Teaches the model to ENRICH a plain line."""
    out: list[tuple] = []
    for note, fig in zip(notes, figs):
        if fig in _ORNAMENT and out and out[-1][0] is not None:
            pm, pd, *pv = out[-1]
            out[-1] = (pm, pd + note[1], *pv)
        else:
            out.append(tuple(note))
    return out
