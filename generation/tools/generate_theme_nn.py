"""Generate a survey of melodies from the trained ThemeGPT and write one score to listen to.

    python -m generation.tools.generate_theme_nn                  # M1-character survey (F major, arch, heroic)
    python -m generation.tools.generate_theme_nn --key C --shape rising --char romance,aria

Baseline model: free melodies conditioned on key + shape + character. No pins / kernel constraint
yet (that is the infilling phase) — so these are "in M1's character", not realizations of the M1
kernel. Output: outputs/sketches/theme_nn/<name>.{mid,musicxml}.
"""
import argparse
from pathlib import Path

from music21 import instrument, pitch as m21pitch

from framework.foundation.paths import SKETCH_OUTPUTS
from framework.foundation.score import mk_part, mk_score, rest_bars, tag
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
            items += tag(block, f"txt:{label}") + rest_bars(1)

    total = sum(float(it[1]) for it in items)
    score = mk_score(f"{args.name}", "4/4", args.key,
                     [mk_part("theme", instrument.Flute(), items, validate_ql=total)],
                     tempo_events=[(0, args.tempo)])
    out_dir = SKETCH_OUTPUTS / "theme_nn"
    out_dir.mkdir(parents=True, exist_ok=True)
    midi = out_dir / f"{args.name}.mid"
    score.write("midi", fp=str(midi))           # MIDI is the point — free durations play fine
    print(f"wrote {midi}")
    try:                                          # notation is best-effort (free durations rarely tile)
        xml = out_dir / f"{args.name}.musicxml"
        score.write("musicxml", fp=str(xml))
        print(f"wrote {xml}")
    except Exception as exc:  # noqa
        print(f"musicxml skipped (inexpressible durations): {exc}")


if __name__ == "__main__":
    main()
