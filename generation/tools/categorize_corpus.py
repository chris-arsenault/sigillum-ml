"""Resumable, idempotent corpus categorization — tag every corpus piece with dramatic/functional
types (one unified VGM-framed taxonomy applied to ALL music: "what setting would this work in?").

The JUDGMENT is the agent's; this tool is the deterministic plumbing around it:

    python -m generation.tools.categorize_corpus --status            # counts: total / tagged / pending
    python -m generation.tools.categorize_corpus --pending 40         # next 40 untagged files + evidence (JSON)
    python -m generation.tools.categorize_corpus --write tags.json    # upsert {path:{primary,supporting}} -> manifest
    python -m generation.tools.categorize_corpus --taxonomy           # the tag set

Resumable: ``--pending`` skips anything already in the manifest, so a stopped run just continues,
and newly-unpacked zips automatically become pending. Idempotent: ``--write`` upserts by path, so
re-tagging a file overwrites its entry rather than duplicating it. The manifest
(``trackers/corpus_categories.jsonl``) is the single source of truth.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "assets" / "raw" / "corpus"
MANIFEST = ROOT / "trackers" / "corpus_categories.jsonl"
_EXTS = {".mid", ".midi"}
# Excluded from the scan: the raw harvest source (not training material), and the quarantine the
# cleaning pass moves misfiled files into. Newly-unpacked VGM zips land outside both and appear.
_EXCLUDE = {"magic_of_midi", "_rejected"}
REJECTED = CORPUS / "_rejected"

# One unified taxonomy, applied to every piece via the VGM lens (a chorale that "would score a
# boss" gets tagged boss + chorale). Tag a primary (1-2 best-fit) and supporting (other flavours).
TAXONOMY = {
    "battle": "active combat, driving",
    "boss": "high-stakes fight, grand/menacing",
    "overworld": "field / map / exploration travel",
    "town": "settlement, peaceful hub, shop",
    "dungeon": "cave / fortress, dangerous exploration",
    "character": "a personality leitmotif",
    "romance": "love, tender, intimate",
    "sad": "loss, lament, tragic",
    "victory": "fanfare, triumph, win jingle",
    "tension": "danger, suspense, unease",
    "title": "title screen, menu, intro",
    "ending": "credits, finale, farewell",
    "ambient": "atmosphere, mystery, eerie",
    "comedy": "quirky, lighthearted, playful",
    "heroic": "noble, grand, resolute",
    "march": "processional, military, steady tread",
    "dance": "waltz / minuet / jig, lilting",
    "chorale": "hymn, sacred, solemn block harmony",
    "fugue": "contrapuntal, intricate, learned",
    "virtuosic": "showy, technical, fast",
    "aria": "lyrical, melody-forward, singable",
    "nocturne": "calm, reflective, night",
}


def scan() -> list[Path]:
    return sorted(
        p for p in CORPUS.rglob("*")
        if p.suffix.lower() in _EXTS
        and not (_EXCLUDE & set(p.relative_to(CORPUS).parts))
    )


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def category_of(path: Path) -> str:
    """The coarse corpus bucket / game folder under corpus/, for context."""
    parts = path.relative_to(CORPUS).parts
    return "/".join(parts[:2]) if parts and parts[0] == "vgm" else (parts[0] if parts else "?")


def load_manifest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text().splitlines():
            line = line.strip()
            if line:
                entry = json.loads(line)
                out[entry["path"]] = entry
    return out


def save_manifest(entries: dict[str, dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text("".join(json.dumps(entries[k]) + "\n" for k in sorted(entries)))


def evidence(path: Path) -> dict:
    """Cheap musical corroboration (no harmony windowing): mode, tempo, density, angularity, range."""
    try:
        import mido
        from framework.analysis.events import events_from_midi, melody_line
        from framework.analysis.tonal import NOTE_NAMES, estimate_key, pc_histogram

        events, total, _barlen = events_from_midi(str(path))
        if not events or total <= 0:
            return {"error": "empty"}
        tonic, mode = estimate_key(pc_histogram(events))
        mel = melody_line(events)
        pitches = [e.midi for e in mel]
        intervals = [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]
        tempo = None
        for track in mido.MidiFile(str(path)).tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = round(mido.tempo2bpm(msg.tempo)); break
            if tempo:
                break
        return {
            "key": f"{NOTE_NAMES[tonic]} {mode}",
            "tempo": tempo,
            "density": round(len(mel) / total, 2),
            "mean_interval": round(sum(intervals) / len(intervals), 1) if intervals else 0,
            "range": (max(pitches) - min(pitches)) if pitches else 0,
            "notes": len(mel),
        }
    except Exception as exc:  # noqa
        return {"error": exc.__class__.__name__}


def cmd_status(_args):
    files = scan()
    done = load_manifest()
    tagged = sum(1 for p in files if rel(p) in done)
    print(f"corpus files: {len(files)}   tagged: {tagged}   pending: {len(files) - tagged}")
    if done:
        from collections import Counter
        tags = Counter(t for e in done.values() for t in e.get("primary", []) + e.get("supporting", []))
        print("tag counts:", dict(tags.most_common()))


def cmd_pending(args):
    done = load_manifest()
    pending = [p for p in scan() if rel(p) not in done][: args.pending]
    for p in pending:
        print(json.dumps({"path": rel(p), "category": category_of(p),
                          "filename": p.name, "evidence": evidence(p)}))


def cmd_write(args):
    data = json.loads(Path(args.write).read_text() if args.write != "-" else sys.stdin.read())
    entries = load_manifest()
    valid = set(TAXONOMY)
    for path, tags in data.items():
        primary = [t for t in tags.get("primary", []) if t in valid]
        supporting = [t for t in tags.get("supporting", []) if t in valid]
        bad = [t for t in tags.get("primary", []) + tags.get("supporting", []) if t not in valid]
        if bad:
            print(f"warning: unknown tags {bad} on {path}", file=sys.stderr)
        prior = entries.get(path, {})
        entries[path] = {"path": path, "category": category_of(ROOT / path),
                         "primary": primary, "supporting": supporting,
                         "verdict": tags.get("verdict", "keep"),
                         "reason": tags.get("reason", ""),
                         "moved_to": prior.get("moved_to"),
                         "evidence": tags.get("evidence", prior.get("evidence", {}))}
    save_manifest(entries)
    print(f"wrote {len(data)} entries; manifest now {len(entries)} total")


def cmd_apply_rejects(_args):
    """Move every reject-verdict file still in place into corpus/_rejected/ (idempotent)."""
    import shutil
    entries = load_manifest()
    moved = 0
    for entry in entries.values():
        if entry.get("verdict") != "reject" or entry.get("moved_to"):
            continue
        src = ROOT / entry["path"]
        if not src.exists():
            continue
        dest = REJECTED / Path(entry["path"]).relative_to(CORPUS.relative_to(ROOT))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        entry["moved_to"] = rel(dest)
        moved += 1
    save_manifest(entries)
    rejects = sum(1 for e in entries.values() if e.get("verdict") == "reject")
    print(f"moved {moved} files to corpus/_rejected/  (total reject-verdict: {rejects})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--pending", type=int, metavar="N")
    parser.add_argument("--write", metavar="JSON")
    parser.add_argument("--apply-rejects", action="store_true")
    parser.add_argument("--taxonomy", action="store_true")
    args = parser.parse_args(argv)
    if args.taxonomy:
        for tag, desc in TAXONOMY.items():
            print(f"  {tag:11s} {desc}")
    elif args.pending is not None:
        cmd_pending(args)
    elif args.write:
        cmd_write(args)
    elif args.apply_rejects:
        cmd_apply_rejects(args)
    else:
        cmd_status(args)


if __name__ == "__main__":
    main()
