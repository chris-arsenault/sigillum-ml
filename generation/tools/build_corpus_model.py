"""Build and persist Markov melody models from corpus specs.

Trains ``MarkovMelodyModel`` and writes a JSON artifact under ``outputs/models/``
(rebuildable, git-ignored). Build a durable named spec, all of them, or an ad-hoc corpus:

    python -m generation.tools.build_corpus_model --list                       # show named specs
    python -m generation.tools.build_corpus_model --spec exotic                # build a named model
    python -m generation.tools.build_corpus_model --all                        # build every named model
    python -m generation.tools.build_corpus_model --sources vgm bach classical # ad-hoc combo
    python -m generation.tools.build_corpus_model --sources fiddle@150 exotic --out my_blend.json

Named specs live in ``generation.theme_gen.model_specs`` (the durable manifest).
Corpus tokens are ``name`` or ``name@N`` (cap to N, sampled): bach / fiddle / folk / vgm /
exotic / classical / arabian / themes / music21 / ingest[:<subdir>] / any category subfolder.
"""
import argparse
from pathlib import Path

from framework.foundation.paths import model_output
from generation.theme_gen.corpus import load_corpus
from generation.theme_gen.model import MarkovMelodyModel
from generation.theme_gen.model_specs import MODEL_SPECS, ModelSpec, get_spec


def build(sources, out=None, *, order=2, beats_per_bar=4.0, min_count=20, sample_seed=0):
    corpus = load_corpus(*sources, sample_seed=sample_seed)
    model = MarkovMelodyModel.from_corpus(
        corpus, order=order, beats_per_bar=beats_per_bar, min_count=min_count
    )
    path = Path(out) if (out and (Path(out).is_absolute() or "/" in str(out))) else model_output(out or "markov.json")
    saved = model.save(path)
    return saved, len(corpus)


def build_spec(name, spec: ModelSpec | None = None):
    spec = spec or get_spec(name)
    return build(
        spec.sources,
        spec.output_name(name),
        order=spec.order,
        beats_per_bar=spec.beats_per_bar,
        min_count=spec.min_count,
        sample_seed=spec.sample_seed,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--spec", help="build a named model spec")
    group.add_argument("--all", action="store_true", help="build every named model spec")
    group.add_argument("--list", action="store_true", help="list named model specs and exit")
    parser.add_argument("--sources", nargs="+", help="ad-hoc corpus tokens (e.g. vgm bach classical fiddle@150)")
    parser.add_argument("--out", default=None, help="output path; bare names land under outputs/models/")
    args = parser.parse_args(argv)

    if args.list:
        for name, spec in MODEL_SPECS.items():
            print(f"  {name:20s} {', '.join(spec.sources):40s} {spec.description}")
        return

    if args.all:
        for name in MODEL_SPECS:
            path, n = build_spec(name)
            print(f"wrote {path}  ({n} melodies)")
        return

    if args.spec:
        path, n = build_spec(args.spec)
        print(f"wrote {path}  ({n} melodies)")
        return

    sources = args.sources or ["music21", "themes"]
    path, n = build(sources, args.out)
    print(f"wrote {path}  ({n} melodies)")


if __name__ == "__main__":
    main()
