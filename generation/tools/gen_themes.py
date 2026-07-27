"""Generate a kernel-constrained theme batch from a DSL-authored kernel (D4).

Load a kernel module by dotted path or .py file, run the diversity pipeline, and write the
audition score (MusicXML + MIDI) and the spread report. The kernel module exposes ``KERNEL``
(a ThemeKernel built with the kernel DSL) or a ``build_kernel()`` callable. Examples:

    python -m generation.tools.gen_themes experiments.s_beloved_kernel --seed 20260613
    python -m generation.tools.gen_themes path/to/my_kernel.py --pool 96 --batch 12 --out outputs/sketches/themes
"""
import argparse
import importlib
import importlib.util
import os
from pathlib import Path

from generation.partitura_bridge import export_score, single_part_score
from generation.project_paths import SKETCH_OUTPUTS, model_output
from generation.theme_gen import (
    GenerationTrace,
    generate_theme_batch,
    render_candidate_report,
    render_run_log,
)
from generation.theme_gen.audition_specs import AUDITIONS, get_audition
from generation.theme_gen.model import MarkovMelodyModel
from generation.theme_gen.report import render_kernel_density


def _is_path(ref: str) -> bool:
    return ref.endswith(".py") or "/" in ref or os.sep in ref


def load_model(ref):
    """Load a trained model by spec name (outputs/models/<name>.json) or file path.

    Returns None (with a warning) if a named model has not been built yet, so generation
    falls back to the in-memory repo-theme default rather than failing.
    """
    if not ref:
        return None
    path = Path(ref) if (str(ref).endswith(".json") or "/" in str(ref)) else model_output(f"{ref}.json")
    if not Path(path).exists():
        print(f"warning: model {ref!r} not found at {path} — "
              f"build it with `python -m generation.tools.build_corpus_model --spec {ref}`; "
              f"using the repo-theme default for now.")
        return None
    return MarkovMelodyModel.load(path)


def load_kernel(ref: str):
    """Load a ThemeKernel from a dotted module path or a .py file path."""
    if _is_path(ref):
        path = Path(ref).resolve()
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(ref)
    if hasattr(module, "KERNEL"):
        return module.KERNEL
    if hasattr(module, "build_kernel"):
        return module.build_kernel()
    raise SystemExit(f"{ref}: module exposes neither KERNEL nor build_kernel()")


def _stem_from_ref(ref: str) -> str:
    return Path(ref).stem if _is_path(ref) else ref.rsplit(".", 1)[-1]


def generate(kernel, *, seed=None, pool=144, batch=12, model=None, trace=None):
    return generate_theme_batch(
        kernel, model=model, pool_size=pool, batch_size=batch, seed=seed, trace=trace
    )


def _label_first_sounding(items, label):
    labelled = [tuple(item) for item in items]
    for index, item in enumerate(labelled):
        if item[0] is not None:
            labelled[index] = (*item, f"txt:{label.replace(' ', '_')}")
            break
    return labelled


def build_score(candidates, kernel, title):
    """Render the candidate spread as labelled blocks, 1-bar rests between (audition layout)."""
    items = []
    for index, candidate in enumerate(candidates, 1):
        items.extend(_label_first_sounding(candidate.items, f"CAND_{index}"))
        items.append((None, kernel.frame.beats_per_bar))
    return single_part_score(
        title=title,
        items=items,
        meter=kernel.frame.meter,
        key=kernel.frame.key,
        tempo=76,
        beats_per_bar=kernel.frame.beats_per_bar,
    )


def build_audition_score(per_model, kernel, title, tempo):
    """One score surveying a kernel across models: each block is ``<model> <n>``."""
    items = []
    for model_name, candidates, _used, _trace in per_model:
        for index, candidate in enumerate(candidates, 1):
            items.extend(
                _label_first_sounding(candidate.items, f"{model_name}_{index}")
            )
            items.append((None, kernel.frame.beats_per_bar))
    return single_part_score(
        title=title,
        items=items,
        meter=kernel.frame.meter,
        key=kernel.frame.key,
        tempo=tempo,
        beats_per_bar=kernel.frame.beats_per_bar,
    )


def _audition_report(name, spec, per_model, kernel):
    lines = [
        f"# Audition: {name}", "", spec.description, "",
        f"Kernel: {spec.kernel}   (seed {spec.seed}, pool {spec.pool}, {spec.batch}/model)",
        render_kernel_density(kernel, batch_size=spec.batch), "",
        "Per model: the Stage-1 generation run-log, then the candidates selected into the "
        "spread (notes are in the MIDI/MusicXML). Spread order is dissimilarity, not merit.", "",
    ]
    for model_name, candidates, used, trace in per_model:
        label = model_name if used else f"{model_name}  [NOT BUILT -> repo-theme fallback]"
        lines.append(f"## {label}")
        lines.append(render_run_log(trace, candidates))
        lines.append("")
    return "\n".join(lines)


def run_audition(name, out=None):
    """Render a named audition: the kernel through every model, into one survey score."""
    spec = get_audition(name)
    kernel = load_kernel(spec.kernel)
    per_model = []
    for model_name in spec.models:
        model = load_model(model_name)
        trace = GenerationTrace()
        candidates = generate(kernel, seed=spec.seed, pool=spec.pool, batch=spec.batch,
                              model=model, trace=trace)
        per_model.append((model_name, candidates, model is not None, trace))

    out_dir = Path(out) if out else (SKETCH_OUTPUTS / "auditions")
    score = build_audition_score(per_model, kernel, f"{name} model survey", spec.tempo)
    xml, midi = export_score(score, out_dir, name)
    report = Path(out_dir) / f"{name}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(_audition_report(name, spec, per_model, kernel), encoding="utf-8")
    print(f"wrote {xml}\nwrote {midi}\nwrote {report}")
    for model_name, candidates, used, _trace in per_model:
        print(f"  {model_name:20s} {len(candidates)} candidates{'' if used else '  (model not built; repo-theme fallback)'}")
    return per_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", nargs="?", help="dotted module path or .py file exposing KERNEL / build_kernel()")
    parser.add_argument("--audition", help="run a named audition (kernel surveyed across models)")
    parser.add_argument("--list-auditions", action="store_true", help="list named auditions and exit")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pool", type=int, default=144)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--out", default=None, help="output directory; default outputs/sketches/themes")
    parser.add_argument("--stem", default=None, help="output filename stem; default derived from the kernel ref")
    parser.add_argument("--model", default="general_tonal",
                        help="trained model: a spec name (outputs/models/<name>.json) or a .json path; "
                             "falls back to the repo themes if not built")
    args = parser.parse_args(argv)

    if args.list_auditions:
        for name, spec in AUDITIONS.items():
            print(f"  {name:16s} {spec.kernel:32s} models: {', '.join(spec.models)}")
        return
    if args.audition:
        run_audition(args.audition, out=args.out)
        return
    if not args.kernel:
        parser.error("a kernel is required (or use --audition / --list-auditions)")

    kernel = load_kernel(args.kernel)
    model = load_model(args.model)
    trace = GenerationTrace()
    candidates = generate(kernel, seed=args.seed, pool=args.pool, batch=args.batch,
                          model=model, trace=trace)
    stem = args.stem or _stem_from_ref(args.kernel)
    out_dir = Path(args.out) if args.out else (SKETCH_OUTPUTS / "themes")

    score = build_score(candidates, kernel, f"{stem} kernel candidates")
    xml, midi = export_score(score, out_dir, stem)
    report = Path(out_dir) / f"{stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_candidate_report(candidates, kernel, trace), encoding="utf-8")
    print(f"wrote {xml}\nwrote {midi}\nwrote {report}")
    return candidates


if __name__ == "__main__":
    main()
