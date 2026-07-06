"""Harvest exotic-scale and strong-melodic-classical melodies from a flat MIDI pool.

Durable two-stage design so work is never lost:

  analyze(path)            -- the EXPENSIVE step: parse + melody-extract -> a small feature
                              record (pitch-class weights, final note, leap fraction). Cached
                              one line per file in a JSONL manifest, appended as we go.
  classify_features(...)   -- CHEAP: apply scale/mode + filename rules to a cached record.

A run appends to the manifest and copies selected files into per-category folders as it
goes (resumable: already-analysed files are skipped). ``--recatalog`` rebuilds the category
folders from the manifest WITHOUT re-parsing, so changing the rules re-buckets 169k files in
seconds. One-off curation utility (not training code).
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import shutil
import warnings

warnings.simplefilter("ignore")

from music21 import pitch as m21pitch

from generation.theme_gen.corpus import ingest_file

SCALES = {
    "major": {0, 2, 4, 5, 7, 9, 11},
    "nat_minor": {0, 2, 3, 5, 7, 8, 10},
    "harm_minor": {0, 2, 3, 5, 7, 8, 11},
    "mel_minor": {0, 2, 3, 5, 7, 9, 11},
    "dorian": {0, 2, 3, 5, 7, 9, 10},
    "mixolydian": {0, 2, 4, 5, 7, 9, 10},
    "lydian": {0, 2, 4, 6, 7, 9, 11},
    "phrygian": {0, 1, 3, 5, 7, 8, 10},
    "phryg_dominant": {0, 1, 4, 5, 7, 8, 10},
    "hungarian_minor": {0, 2, 3, 6, 7, 8, 11},
    "double_harmonic": {0, 1, 4, 5, 7, 8, 11},
    "japanese_in": {0, 1, 5, 7, 8},
    "hirajoshi": {0, 2, 3, 7, 8},
    "whole_tone": {0, 2, 4, 6, 8, 10},
}
# Phrygian is a diatonic church mode (one ♭2 from natural minor) and was wildly
# over-detected on ordinary pop/minor tunes, so it is NOT exotic — it lives with the
# western modes. Genuine exotic = scales with augmented 2nds / no perfect 5th.
EXOTIC = {"phryg_dominant", "hungarian_minor", "double_harmonic", "japanese_in",
          "hirajoshi", "whole_tone"}
WESTERN = {"major", "nat_minor", "harm_minor", "mel_minor", "dorian", "mixolydian",
           "lydian", "phrygian"}

# Characteristic scale degrees (relative to tonic) a melody must actually USE to earn the
# exotic label — this is what stops a chromatic pop tune coincidentally fitting.
EXOTIC_SIGNATURE = {
    "phryg_dominant": (1, 4),          # ♭2 + M3 (the Hijaz augmented 2nd)
    "double_harmonic": (1, 4, 8),      # ♭2 + M3 + ♭6
    "hungarian_minor": (3, 6),         # ♭3 + ♯4
    "japanese_in": (1, 8),             # ♭2 + ♭6
    "hirajoshi": (3, 8),               # ♭3 + ♭6
    "whole_tone": (2, 4, 6, 8, 10),    # the whole-tone collection (no perfect 5th)
}

MIN_NOTES = 24
EXOTIC_FIT = 0.93
EXOTIC_MARGIN = 0.07
EXOTIC_SIG_WEIGHT = 0.04
EXOTIC_MIN_TONIC_WEIGHT = 0.10
CLASSICAL_MAX_BIG_LEAP_FRAC = 0.20

_CLASSICAL_RX = re.compile(
    r"bach|mozart|beethoven|chopin|schubert|brahms|handel|haydn|vivaldi|tchaik|debussy|liszt|"
    r"mendelssohn|schumann|dvorak|grieg|ravel|satie|rachmanin|wagner|verdi|puccini|scarlatti|"
    r"telemann|purcell|faure|bizet|saint.?saens|paganini|albinoni|pachelbel|rossini|"
    r"holst|elgar|sibelius|prokofiev|mahler|bruckner|clementi|"
    r"sonatina|sonata|symphony|concerto|nocturne|etude|\bprelude|fugue|minuet|menuet|waltz|"
    r"mazurka|polonaise|rhapsod|gavotte|sarabande|\bgigue|\bopus\b|\bop\.? ?\d|canon|fur.?elise|"
    r"moonlight|ode.?to.?joy|clair.?de|ave.?maria|requiem|adagio|andante|cantata|impromptu|"
    r"barcarolle|berceuse|invention|bourree|toccata|partita",
    re.I,
)


def _best_scale_at(weights, total, tonics):
    """Best (fit, scale, tonic) over candidate tonics, in priority order (final note first)."""
    best = (0.0, None, None)
    for tonic in tonics:
        for name, intervals in SCALES.items():
            scale_pcs = {(tonic + i) % 12 for i in intervals}
            fit = sum(c for pc, c in weights.items() if pc in scale_pcs) / total
            if fit > best[0]:
                best = (fit, name, tonic)
    return best


def analyze(path):
    """Expensive: parse + extract melody -> cached feature record, or None if unusable."""
    line = ingest_file(path)
    if not line:
        return None
    midis = [int(m21pitch.Pitch(name).midi) for name, *_ in line if name is not None]
    if len(midis) < MIN_NOTES:
        return None
    weights = collections.Counter(m % 12 for m in midis)
    intervals = [abs(midis[i + 1] - midis[i]) for i in range(len(midis) - 1)]
    blf = sum(1 for v in intervals if v > 12) / len(intervals) if intervals else 1.0
    return {"n": len(midis), "final": midis[-1] % 12,
            "pcw": {str(pc): c for pc, c in weights.items()}, "blf": round(blf, 3)}


def classify_features(feat, basename):
    """Cheap: (label, scale, fit, west_fit) from a cached feature record, or None."""
    if feat is None:
        return None
    weights = collections.Counter({int(pc): c for pc, c in feat["pcw"].items()})
    total = sum(weights.values())
    if total == 0:
        return None
    final = feat["final"]
    ranked = [pc for pc, _ in weights.most_common()]
    tonics = list(dict.fromkeys([final] + ranked[:2]))

    fit, scale, tonic = _best_scale_at(weights, total, tonics)
    west_fit = max(
        sum(c for pc, c in weights.items() if pc in {(tonic + i) % 12 for i in SCALES[n]}) / total
        for n in WESTERN
    )

    if (
        scale in EXOTIC
        and fit >= EXOTIC_FIT
        and fit > west_fit + EXOTIC_MARGIN
        and tonic == final
        and weights[tonic] / total >= EXOTIC_MIN_TONIC_WEIGHT
        and all(weights[(tonic + deg) % 12] / total >= EXOTIC_SIG_WEIGHT
                for deg in EXOTIC_SIGNATURE[scale])
    ):
        return ("exotic", scale, round(fit, 2), round(west_fit, 2))

    if (
        _CLASSICAL_RX.search(basename)
        and len(weights) >= 6
        and feat["blf"] <= CLASSICAL_MAX_BIG_LEAP_FRAC
    ):
        return ("classical", scale, round(fit, 2), round(west_fit, 2))
    return None


def classify(path):
    """Convenience: analyze + classify one path (used for validation)."""
    return classify_features(analyze(path), os.path.basename(path))


def _dest(out, label, scale):
    return os.path.join(out, label, scale)


def _copy(path, dest):
    if os.path.exists(path):
        os.makedirs(dest, exist_ok=True)
        shutil.copy2(path, os.path.join(dest, os.path.basename(path)))


def recatalog(manifest, out, max_classical):
    """Rebuild category folders from the manifest WITHOUT re-parsing (rule changes / recovery)."""
    for label in ("exotic", "classical"):
        shutil.rmtree(os.path.join(out, label), ignore_errors=True)
    by_scale = collections.Counter()
    n_exotic = n_classical = 0
    with open(manifest, encoding="utf-8") as fh:
        for raw in fh:
            rec = json.loads(raw)
            if "pcw" not in rec:
                continue
            label = classify_features(rec, os.path.basename(rec["path"]))
            if label is None:
                continue
            kind, scale = label[0], label[1]
            if kind == "exotic":
                n_exotic += 1
            elif n_classical < max_classical:
                n_classical += 1
            else:
                continue
            by_scale[(kind, scale)] += 1
            _copy(rec["path"], _dest(out, kind, scale))
    print(f"recatalog: exotic={n_exotic} classical={n_classical}")
    for (kind, scale), c in by_scale.most_common():
        print(f"  {kind:9s} {scale:16s}: {c}")


def _analysed_paths(manifest):
    done = set()
    if os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as fh:
            for raw in fh:
                try:
                    done.add(json.loads(raw)["path"])
                except Exception:
                    pass
    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="assets/raw/corpus/magic_of_midi/MIDI")
    parser.add_argument("--out", default="assets/raw/corpus")
    parser.add_argument("--manifest", default="outputs/magic_of_midi_manifest.jsonl")
    parser.add_argument("--max-classical", type=int, default=500)
    parser.add_argument("--stop-at", type=int, default=0, help="stop scanning after N selected (0=whole pool)")
    parser.add_argument("--limit-scan", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--recatalog", action="store_true")
    args = parser.parse_args(argv)

    if args.recatalog:
        recatalog(args.manifest, args.out, args.max_classical)
        return

    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    done = _analysed_paths(args.manifest)
    files = [f for f in glob.glob(os.path.join(args.pool, "*")) if f.lower().endswith((".mid", ".midi"))]
    random.Random(args.seed).shuffle(files)
    if args.limit_scan:
        files = files[: args.limit_scan]

    n_exotic = n_classical = scanned = errors = skipped = 0
    by_scale = collections.Counter()
    with open(args.manifest, "a", encoding="utf-8") as mf:
        for path in files:
            if path in done:
                skipped += 1
                continue
            scanned += 1
            try:
                feat = analyze(path)
            except Exception:
                errors += 1
                feat = None
            rec = {"path": path}
            if feat:
                rec.update(feat)
            label = classify_features(feat, os.path.basename(path)) if feat else None
            if label is not None:
                kind, scale = label[0], label[1]
                if kind == "exotic":
                    rec["label"], rec["scale"] = kind, scale
                    _copy(path, _dest(args.out, kind, scale))
                    n_exotic += 1
                    by_scale[(kind, scale)] += 1
                elif n_classical < args.max_classical:
                    rec["label"], rec["scale"] = kind, scale
                    _copy(path, _dest(args.out, kind, scale))
                    n_classical += 1
                    by_scale[(kind, scale)] += 1
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            if args.stop_at and (n_exotic + n_classical) >= args.stop_at:
                break
            if scanned % 5000 == 0:
                print(f"  scanned {scanned} (skipped {skipped}) | exotic {n_exotic} | classical {n_classical}", flush=True)

    print(f"\nscanned={scanned} skipped={skipped} errors={errors}  selected: exotic={n_exotic} classical={n_classical}")
    for (kind, scale), c in by_scale.most_common():
        print(f"  {kind:9s} {scale:16s}: {c}")


if __name__ == "__main__":
    main()
