"""Generate a survey of melodies from the trained ThemeGPT and write one score to listen to.

    python -m generation.tools.generate_theme_nn                  # M1-character survey (F major, arch, heroic)
    python -m generation.tools.generate_theme_nn --key C --shape rising --char romance,aria

Baseline model: free melodies conditioned on key + shape + character. No pins / kernel constraint
yet (that is the infilling phase) — so these are "in M1's character", not realizations of the M1
kernel. Output: outputs/sketches/theme_nn/<name>.{mid,musicxml}.
"""
import argparse
from pathlib import Path

from music21 import pitch as m21pitch

from generation.partitura_bridge import export_score, single_part_score
from generation.project_paths import SKETCH_OUTPUTS
from generation.theme_nn.generate import generate_melody, load_model

CHECKPOINT = Path(__file__).resolve().parents[2] / "outputs" / "models" / "theme_nn"
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _key_pc(name: str) -> int:
    pc = _PC[name[0].upper()]
    if len(name) > 1 and name[1] in "#b-":
        pc += 1 if name[1] == "#" else -1
    return pc % 12


def _score_items(melody):
    return [(None if m is None else m21pitch.Pitch(midi=m).nameWithOctave, round(d, 4)) for m, d in melody]


def _label_first_sounding(items, label):
    labelled = [tuple(item) for item in items]
    for index, item in enumerate(labelled):
        if item[0] is not None:
            labelled[index] = (*item, f"txt:{label.replace(' ', '_')}")
            break
    return labelled


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", default="F")
    parser.add_argument("--mode", default="major")
    parser.add_argument("--shape", default="arch")
    parser.add_argument("--char", default="heroic,character")
    parser.add_argument("--max-events", type=int, default=160)
    parser.add_argument("--tempo", type=int, default=76)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--name", default="m1_theme_nn_survey")
    args = parser.parse_args(argv)

    model, vocab = load_model(CHECKPOINT)
    key = (_key_pc(args.key), args.mode)
    base_char = [c for c in args.char.split(",") if c]

    # A few "model options": temperature sweep on the M1 conditioning, plus two character variants.
    options = [
        (f"{args.key} {args.shape} {'+'.join(base_char)}  t0.7", base_char, 0.7),
        (f"{args.key} {args.shape} {'+'.join(base_char)}  t0.9", base_char, 0.9),
        (f"{args.key} {args.shape} {'+'.join(base_char)}  t1.1", base_char, 1.1),
        (f"{args.key} {args.shape} romance+aria  t0.9", ["romance", "aria"], 0.9),
        (f"{args.key} {args.shape} battle+tension  t0.9", ["battle", "tension"], 0.9),
    ]

    items = []
    for i, (label, character, temp) in enumerate(options):
        print(f"generating: {label}")
        melody = generate_melody(model, vocab, key=key, shape=args.shape, character=character,
                                 temperature=temp, top_k=40, max_events=args.max_events,
                                 seed=args.seed + i)
        block = _score_items(melody)
        if block:
            items.extend(_label_first_sounding(block, label))
            items.append((None, 4.0))

    score = single_part_score(
        title=args.name,
        items=items,
        meter="4/4",
        key=args.key,
        tempo=args.tempo,
        beats_per_bar=4.0,
    )
    out_dir = SKETCH_OUTPUTS / "theme_nn"
    xml, midi = export_score(score, out_dir, args.name)
    print(f"wrote {midi}")
    print(f"wrote {xml}")


if __name__ == "__main__":
    main()
