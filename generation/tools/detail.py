"""Detail (fractalize) a coarse monophonic line with the trained v2 diffusion detailer.

    python -m generation.tools.detail                    # eval on corpus windows (metrics) + detail the RENN theme

Coarse input is key-normalised to C, rendered at flat velocity; the model adds detail conditioned on
density + character + harmony with classifier-free guidance, anchoring the given structural notes.
Renders coarse->detailed pairs to outputs/sketches/detailer/.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from music21 import instrument, note as m21note, pitch as m21p, stream, tempo

from generation.project_paths import SKETCH_OUTPUTS
from generation.fractal.dataset import CHAR_IDX, HARM_IDX, melody_windows
from generation.fractal.model import Denoiser, discretize, make_schedule, sample
from generation.fractal.roll import C, FLAT_VEL, P, VEL, melody_to_roll, roll_to_melody, strip_ornaments

CHECKPOINT = Path(__file__).resolve().parents[2] / "outputs" / "models" / "detailer"
SCALE_C = {0, 2, 4, 5, 7, 9, 11}
MAXC, MAXH = 6, 4


def load_model(checkpoint_dir: Path = CHECKPOINT):
    ck = torch.load(Path(checkpoint_dir) / "model.pt", map_location="cpu", weights_only=False)
    model = Denoiser(width=ck["config"]["width"], depth=ck["config"]["depth"])
    model.load_state_dict(ck["model"]); model.eval()
    return model, make_schedule(steps=ck["config"]["diffusion_steps"])


# NOTE: full-cell anchoring re-imposes the coarse on every step → output == input (no detail). Until
# onset-only anchoring lands, preservation comes from the coarse-roll CONDITIONING (~83%); anchor=False.
@torch.no_grad()
def detail(model, sched, coarse_notes, *, character=(), harmony=(), density=2.5,
           guidance=2.0, anchor=False, seed=0):
    """coarse line (already in C, (midi|None,dur[,vel])) -> detailed (midi,dur,vel) in C."""
    torch.manual_seed(seed)
    coa = torch.tensor(melody_to_roll(coarse_notes, flat_vel=FLAT_VEL)[None] * 2 - 1)
    char = torch.tensor([[CHAR_IDX.get(c, 0) for c in list(character)[:MAXC]] + [0] * MAXC][:1])[:, :MAXC]
    harm = torch.tensor([[HARM_IDX.get(h, HARM_IDX["?"]) for h in list(harmony)[:MAXH]] + [0] * MAXH][:1])[:, :MAXH]
    dens = torch.tensor([float(density)])
    anc = None
    if anchor:                                       # anchor pitch/rest/onset of the given notes, not velocity
        mask = torch.zeros(1, C, 1); mask[0, : P + 2] = 1.0
        anc = (mask, coa)
    x = sample(model, coa, char, harm, dens, sched, guidance=guidance, anchor=anc)
    return roll_to_melody(discretize(x[0]).numpy())


def _stats(notes):
    midis = [m for m, _d, _v in notes if m is not None]
    vels = [v for _m, _d, v in notes if _m is not None]
    inkey = sum(1 for m in midis if m % 12 in SCALE_C) / max(len(midis), 1) * 100
    return len(midis), inkey, (min(vels), max(vels)) if vels else (0, 0)


def _render(pairs, path, ks="C"):
    s = stream.Score(); part = stream.Part(); part.insert(0, instrument.Flute()); part.insert(0, tempo.MetronomeMark(number=84))
    for label, notes in pairs:
        part.append(m21note.Rest(quarterLength=2))
        for m, d, v in notes:
            if m is None:
                part.append(m21note.Rest(quarterLength=max(d, 1 / 12)))
            else:
                nt = m21note.Note(max(36, min(96, m)), quarterLength=max(d, 1 / 12)); nt.volume.velocity = max(1, min(127, v))
                part.append(nt)
    s.insert(0, part); s.write("midi", fp=str(path))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--density", type=float, default=2.5)
    p.add_argument("--guidance", type=float, default=2.5)
    a = p.parse_args(argv)
    model, sched = load_model()
    out = SKETCH_OUTPUTS / "detailer"; out.mkdir(parents=True, exist_ok=True)

    # 1) corpus windows (in C): ornament-strip -> detail; tests the trained task on real in-key material
    wins = melody_windows(limit=300)
    pairs = []
    print("=== corpus windows (ornament-strip -> detail) ===")
    for i in range(5):
        w = wins[i * 11]
        coarse = strip_ornaments(w.notes, w.figs)
        det = detail(model, sched, coarse, character=[c for c in [] ], harmony=[],
                     density=a.density, guidance=a.guidance, seed=i)
        cn = sum(1 for m, *_ in coarse if m is not None)
        dn, ik, vr = _stats(det)
        struct = sum(1 for cm, *_ in coarse if cm is not None
                     and any(dm == cm for dm, *_ in det)) / max(cn, 1) * 100
        print(f"  ex {i}: coarse {cn}n -> detailed {dn}n | in-key {ik:.0f}% | vel {vr} | struct-kept {struct:.0f}%")
        pairs += [(f"{i} coarse", [(m, d, FLAT_VEL) for m, d, *_ in coarse]), (f"{i} detailed", det)]
    _render(pairs, out / "detailed.mid")
    print("wrote", out / "detailed.mid")

    # 2) the RENN theme (transpose Eb -> C, detail, -> back)
    from symphony.materials.themes import RENN_FULL
    def to_items(t):
        o = []
        for it in t:
            pp, d = it[0], float(it[1])
            o.append((None, d, 0) if pp is None else (int(m21p.Pitch(pp[-1] if isinstance(pp, list) else pp).midi), d, FLAT_VEL))
        return o
    renn = to_items(RENN_FULL); TONIC = 3   # Eb
    rc = [(None if m is None else m - TONIC, d, v) for m, d, v in renn]   # -> C
    # window into 16-beat chunks
    chunks, cur, acc = [], [], 0.0
    for it in rc:
        cur.append(it); acc += it[1]
        if acc >= 16 - 1e-6: chunks.append(cur); cur, acc = [], 0.0
    if cur: chunks.append(cur)
    det = []
    for ci, ch in enumerate(chunks):
        d = detail(model, sched, ch, character=["heroic"], harmony=[], density=a.density, guidance=a.guidance, seed=ci)
        det += [(None if m is None else m + TONIC, dd, v) for m, dd, v in d]   # -> Eb
    dn, _ik, vr = _stats(det)
    rennEb = [(m, d, 80) for m, d, *_ in renn]
    print(f"=== RENN: {sum(1 for m,*_ in renn if m is not None)}n -> detailed {dn}n | vel {vr} ===")
    _render([("RENN original", rennEb), ("RENN detailed", det)], out / "renn_v2.mid")
    print("wrote", out / "renn_v2.mid")


if __name__ == "__main__":
    main()
