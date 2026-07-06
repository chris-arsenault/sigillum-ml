"""Fill the M1 kernel with the trained infilling model and render the result.

    python -m generation.tools.generate_infill            # fill M1 (head pinned b1, apex pinned b14), write MIDI

The kernel's pins are anchors; the model fills the free spans and lands the pins (tiling). Output:
outputs/sketches/theme_nn/m1_infill.mid.
"""
import argparse
from pathlib import Path

from music21 import instrument, pitch as m21pitch

from framework.foundation.paths import SKETCH_OUTPUTS
from framework.foundation.score import mk_part, mk_score, rest_bars, tag
from generation.theme_nn.generate_infill import fill, load_model, to_melody
from generation.theme_nn.representation import encode

CHECKPOINT = Path(__file__).resolve().parents[2] / "outputs" / "models" / "theme_nn_infill"
KEY = (5, "major")   # F major (the M1 theme)


def _anchor(items, key):
    midi_items = [(int(m21pitch.Pitch(p).midi), d) for p, d in items]
    return [(0 if e.kind == "rest" else e.degree, e.alteration, e.octave, e.duration)
            for e in encode(midi_items, key)]


def m1_spec():
    head = _anchor([("C4", 0.5), ("C4", 0.5), ("F4", 1.5), ("E4", 0.5), ("F4", 1.0)], KEY)  # bar 1
    apex = _anchor([("F4", 0.5), ("E5", 2.0)], KEY)                                          # bar 14
    # 16 bars, 4/4 = 64 beats. head = beats 0-4; apex onset beat 52; apex spans 52-54.5.
    return {
        "key": KEY, "shape": "arch", "character": ["heroic", "character"],
        "harmony": ["I", "I", "I", "V", "V", "V", "V", "I", "I", "I", "I", "V", "I", "I", "I", "I"],
        "segments": [("anchor", head), ("gap", 48.0), ("anchor", apex), ("gap", 9.5)],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp", type=float, default=0.9)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--tempo", type=int, default=76)
    args = parser.parse_args(argv)

    model, vocab = load_model(CHECKPOINT)
    spec = m1_spec()
    items = []
    for i in range(args.n):
        melody = fill(model, vocab, spec, temperature=args.temp, seed=20260615 + i)
        block = [(None if m is None else m21pitch.Pitch(midi=max(36, min(96, m))).nameWithOctave, round(d, 4))
                 for m, d in to_melody(melody, KEY)]
        total_beats = sum(d for _m, d in to_melody(melody, KEY))
        print(f"fill {i}: {len(block)} events, {total_beats:.1f} beats "
              f"(kernel wants 64) — head pinned, apex pinned")
        if block:
            items += tag(block, f"txt:M1 infill {i}  t{args.temp}") + rest_bars(1)

    total = sum(float(it[1]) for it in items)
    score = mk_score("m1_infill", "4/4", "F",
                     [mk_part("theme", instrument.Flute(), items, validate_ql=total)],
                     tempo_events=[(0, args.tempo)])
    out_dir = SKETCH_OUTPUTS / "theme_nn"; out_dir.mkdir(parents=True, exist_ok=True)
    midi = out_dir / "m1_infill.mid"
    score.write("midi", fp=str(midi))
    print(f"wrote {midi}")


if __name__ == "__main__":
    main()
