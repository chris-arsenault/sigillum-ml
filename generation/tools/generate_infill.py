"""Fill the M1 kernel with the trained infilling model and render the result.

    python -m generation.tools.generate_infill            # fill M1 (head pinned b1, apex pinned b14), write MIDI

The kernel's pins are anchors; the model fills the free spans and lands the pins (tiling). Output:
outputs/sketches/theme_nn/m1_infill.mid.
"""
import argparse
from pathlib import Path

from music21 import pitch as m21pitch

from generation.partitura_bridge import export_score, single_part_score
from generation.project_paths import SKETCH_OUTPUTS
from generation.theme_nn.generate_infill import fill, load_model, to_melody
from generation.theme_nn.representation import encode

CHECKPOINT = Path(__file__).resolve().parents[2] / "outputs" / "models" / "theme_nn_infill"
KEY = (5, "major")   # F major (the M1 theme)


def _anchor(items, key):
    midi_items = [(int(m21pitch.Pitch(p).midi), d) for p, d in items]
    return [(0 if e.kind == "rest" else e.degree, e.alteration, e.octave, e.duration)
            for e in encode(midi_items, key)]


def _label_first_sounding(items, label):
    labelled = [tuple(item) for item in items]
    for index, item in enumerate(labelled):
        if item[0] is not None:
            labelled[index] = (*item, f"txt:{label.replace(' ', '_')}")
            break
    return labelled


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
            items.extend(
                _label_first_sounding(block, f"M1_infill_{i}_t{args.temp}")
            )
            items.append((None, 4.0))

    score = single_part_score(
        title="m1_infill",
        items=items,
        meter="4/4",
        key="F",
        tempo=args.tempo,
        beats_per_bar=4.0,
    )
    out_dir = SKETCH_OUTPUTS / "theme_nn"
    _xml, midi = export_score(score, out_dir, "m1_infill")
    print(f"wrote {midi}")


if __name__ == "__main__":
    main()
