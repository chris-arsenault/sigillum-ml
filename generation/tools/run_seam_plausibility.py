"""Probe structural-context V4 on real Partitura workflow candidates.

The study uses three Movement IV eight-measure seams. For each seam the real
Ruby composition workflow receives four source-patch candidates: the authentic
continuation, two musically plausible alternatives, and one gross discontinuity
used only as a positive control. Ruby compiles and normally exports every
candidate, then Python scores only Ruby's canonical observations with the
frozen V4 checkpoint.

This is a deliberately small plausibility test, not a trained critic benchmark.
It directly tests whether the learned residual improves candidate ordering over
the boundary-profile baseline on actual composition choices.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from generation.composition import (
    ORIGINAL_CANDIDATE_ID,
    CandidateProposal,
    ProposalRequest,
    ProposalResponse,
    ScoreObservation,
    SelectionRequest,
    SelectionResponse,
)
from generation.composition.structural_context import (
    StructuralContextSpec,
    StructuralSeamScorer,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments" / "whole_score" / "seam_plausibility_v1"
STUDY_ROOT = EXPERIMENT / "study"
TEMPLATE = STUDY_ROOT / "template.rb"
CONTEXT_SPEC = (
    ROOT / "experiments" / "whole_score" / "structural_context_v4" / "experiment.json"
)
CHECKPOINT = (
    ROOT
    / "outputs"
    / "experiments"
    / "whole_score"
    / "structural_context_v4"
    / "checkpoint.pt"
)
OUTPUT_ROOT = ROOT / "outputs" / "experiments" / "whole_score" / "seam_plausibility_v1"
REVIEW_ROOT = ROOT / "outputs" / "reviews" / "whole_score" / "seam_plausibility_v1"
PARTITURA = ROOT.parent / "sigillum-library" / "partitura" / "bin" / "partitura"

CASES = (
    {
        "case_id": "storyteller",
        "title": "Movement IV plain telling",
        "key": "F",
        "meter": 'meter "7/8", beat_pattern: [3, 2, 2]',
    },
    {
        "case_id": "lament",
        "title": "Movement IV slow narrator",
        "key": "F",
        "meter": 'meter "4/4"',
    },
    {
        "case_id": "pulse",
        "title": "Movement IV driving return",
        "key": "Ab",
        "meter": 'meter "7/8", beat_pattern: [3, 2, 2]',
    },
)

CANDIDATES = (
    {
        "name": "coherent_a",
        "label": "coherent",
        "description": (
            "Authentic continuation copied from the current Movement IV "
            "source excerpt."
        ),
    },
    {
        "name": "coherent_b",
        "label": "coherent",
        "description": "Hand-written alternate continuation preserving the source direction.",
    },
    {
        "name": "hard_alternative",
        "label": "plausible",
        "description": (
            "Boundary-matched, musically plausible alternate continuation "
            "preserving the source register and accompaniment."
        ),
    },
    {
        "name": "discontinuous",
        "label": "discontinuous",
        "description": (
            "Gross key, register, and continuity break retained as a positive "
            "control for seam sensitivity."
        ),
    },
)


class SeamPlausibilityError(RuntimeError):
    """The probe could not complete against the real workflow."""


@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    name: str
    label: str
    description: str
    learned_mean: float
    learned_tenth_percentile: float
    learned_minimum: float
    boundary_mean: float
    residual_mean: float
    adjacency_count: int


def _partitura(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("ruby", str(PARTITURA), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )


def _base_source(case: dict[str, str]) -> str:
    text = TEMPLATE.read_text(encoding="utf-8")
    case_root = STUDY_ROOT / case["case_id"]
    anchor = (case_root / "anchor.rbfrag").read_text(encoding="utf-8")
    continuation_color = (case_root / "context.rbfrag").read_text(
        encoding="utf-8"
    )
    replacements = (
        ("CASE_TITLE", case["title"], False),
        ("CASE_ID", f"seam_{case['case_id']}", False),
        ("CASE_KEY", case["key"], False),
        ("METER_DECLARATION", case["meter"], True),
        ("      # ANCHOR_CONTENT", anchor.rstrip(), True),
        (
            "      # CONTINUATION_COLOR",
            continuation_color.rstrip(),
            True,
        ),
    )
    for before, after, unique in replacements:
        count = text.count(before)
        if count == 0 or (unique and count != 1):
            raise SeamPlausibilityError(
                f"{case['case_id']}: invalid template marker {before!r}"
            )
        text = text.replace(before, after)
    return text


def _make_patch(base_text: str, fragment: str, name: str) -> str:
    marker = "      # CANDIDATE_CONTENT"
    if base_text.count(marker) != 1:
        raise SeamPlausibilityError(
            f"{name}: base study must contain one candidate marker"
        )
    variant_text = base_text.replace(marker, fragment.rstrip())
    body = "".join(
        difflib.unified_diff(
            base_text.splitlines(keepends=True),
            variant_text.splitlines(keepends=True),
            fromfile=f"a/{name}",
            tofile=f"b/{name}",
        )
    )
    if not body:
        raise SeamPlausibilityError(f"{name}: candidate is identical to the base")
    return f"diff --git a/{name} b/{name}\n{body}"


def _selection_request(
    directory: Path, case: dict[str, str]
) -> tuple[
    SelectionRequest,
    dict[str, dict[str, str]],
    dict[str, Path],
]:
    case_id = case["case_id"]
    source = directory / f"seam_{case_id}.rb"
    trajectory = directory / f"{case_id}-trajectory.jsonl"
    proposals = directory / f"{case_id}-proposals.json"
    base_text = _base_source(case)
    source.write_text(base_text, encoding="utf-8")

    observed = _partitura(
        "observe", str(source), "--trajectory", str(trajectory), "--no-export"
    )
    if observed.returncode != 0:
        raise SeamPlausibilityError(
            f"{case_id}: observe failed: {observed.stderr.strip()}"
        )
    request = ProposalRequest.from_json(observed.stdout)
    if request.action.get("target_path") != "span:continuation":
        raise SeamPlausibilityError(
            f"{case_id}: scheduler targeted {request.action.get('target_path')}"
        )

    metadata: dict[str, dict[str, str]] = {}
    proposals_for_case = []
    for candidate in CANDIDATES:
        fragment = (
            STUDY_ROOT / case_id / f"{candidate['name']}.rbfrag"
        ).read_text(encoding="utf-8")
        candidate_id = f"candidate:{case_id}:{candidate['name']}"
        proposals_for_case.append(
            CandidateProposal.inline(
                request,
                source_patch=_make_patch(base_text, fragment, source.name),
                description=candidate["description"],
                candidate_id=candidate_id,
            )
        )
        metadata[candidate_id] = candidate

    response = ProposalResponse.create(
        request,
        producer="seam-plausibility-probe",
        candidates=tuple(proposals_for_case),
    )
    proposals.write_text(response.to_json(), encoding="utf-8")
    evaluated = _partitura(
        "evaluate",
        str(source),
        "--trajectory",
        str(trajectory),
        "--proposals",
        str(proposals),
    )
    if evaluated.returncode != 0:
        raise SeamPlausibilityError(
            f"{case_id}: evaluate failed: {evaluated.stderr.strip()}"
        )
    return (
        SelectionRequest.from_json(evaluated.stdout),
        metadata,
        {
            "source": source,
            "trajectory": trajectory,
            "proposals": proposals,
        },
    )


def _score_case(
    *,
    directory: Path,
    case: dict[str, str],
    scorer: StructuralSeamScorer,
    review_root: Path | None = None,
) -> dict[str, Any]:
    selection, metadata, workflow_paths = _selection_request(directory, case)
    observations = selection.to_dict()["candidate_observations"]
    rejected = sorted(set(selection.candidate_ids) - set(observations))
    scored = []
    for candidate_id, raw_observation in observations.items():
        signal = scorer.score(ScoreObservation.from_dict(raw_observation))
        candidate = metadata[candidate_id]
        scored.append(
            ScoredCandidate(
                candidate_id=candidate_id,
                name=candidate["name"],
                label=candidate["label"],
                description=candidate["description"],
                learned_mean=signal.learned_mean,
                learned_tenth_percentile=signal.learned_tenth_percentile,
                learned_minimum=signal.learned_minimum,
                boundary_mean=signal.boundary_mean,
                residual_mean=signal.residual_mean,
                adjacency_count=signal.adjacency_count,
            )
        )
    learned = sorted(scored, key=lambda item: item.learned_mean, reverse=True)
    boundary = sorted(scored, key=lambda item: item.boundary_mean, reverse=True)
    result = {
        "case_id": case["case_id"],
        "title": case["title"],
        "candidate_count": len(selection.candidate_ids),
        "exported_count": len(observations),
        "rejected_candidates": rejected,
        "candidates": [asdict(item) for item in scored],
        "learned_ranking": [item.name for item in learned],
        "boundary_ranking": [item.name for item in boundary],
    }
    if review_root is not None:
        result["review"] = _build_review(
            directory=directory,
            review_root=review_root,
            case=case,
            selection=selection,
            workflow_paths=workflow_paths,
        )
    return result


def _build_review(
    *,
    directory: Path,
    review_root: Path,
    case: dict[str, str],
    selection: SelectionRequest,
    workflow_paths: dict[str, Path],
) -> dict[str, Any]:
    case_id = case["case_id"]
    selection_path = directory / f"{case_id}-selection.json"
    response = SelectionResponse.create(
        selection,
        producer="seam-plausibility-recording-policy",
        selected_candidate_id=ORIGINAL_CANDIDATE_ID,
        reason=(
            "Keep the open study source unchanged while retaining all real "
            "candidates for blinded human review."
        ),
    )
    selection_path.write_text(response.to_json(), encoding="utf-8")
    stepped = _partitura(
        "step",
        str(workflow_paths["source"]),
        "--trajectory",
        str(workflow_paths["trajectory"]),
        "--proposals",
        str(workflow_paths["proposals"]),
        "--selection",
        str(selection_path),
        "--trajectory-origin",
        "agent",
        "--trajectory-quality",
        "medium",
    )
    if stepped.returncode != 0:
        raise SeamPlausibilityError(
            f"{case_id}: step failed: {stepped.stderr.strip()}"
        )
    transition_id = json.loads(stepped.stdout)["transition"]["transition_id"]
    reviews = review_root / "private" / "reviews.jsonl"
    bundles = review_root / "bundles"
    reviewed = _partitura(
        "review",
        "--trajectory",
        str(workflow_paths["trajectory"]),
        "--reviews",
        str(reviews),
        "--output",
        str(bundles),
        "--transition",
        transition_id,
        "--candidate",
        f"candidate:{case_id}:coherent_a",
        "--against",
        f"candidate:{case_id}:hard_alternative",
        "--scale",
        "seam",
        "--criterion",
        "coherence",
        "--seed",
        "seam-plausibility-v1",
    )
    if reviewed.returncode != 0:
        raise SeamPlausibilityError(
            f"{case_id}: review failed: {reviewed.stderr.strip()}"
        )
    public = json.loads(reviewed.stdout)
    bundle = Path(public.pop("bundle"))
    _render_preview(bundle / "A.mid", bundle / "A.wav")
    _render_preview(bundle / "B.mid", bundle / "B.wav")
    return {
        "review_id": public["review_id"],
        "title": case["title"],
        "bundle": str(bundle.relative_to(review_root)),
        "audio": {
            "A": str((bundle / "A.wav").relative_to(review_root)),
            "B": str((bundle / "B.wav").relative_to(review_root)),
        },
        "midi": {
            "A": str((bundle / "A.mid").relative_to(review_root)),
            "B": str((bundle / "B.mid").relative_to(review_root)),
        },
        "musicxml": {
            "A": str((bundle / "A.musicxml").relative_to(review_root)),
            "B": str((bundle / "B.musicxml").relative_to(review_root)),
        },
    }


def _render_preview(midi_path: Path, wav_path: Path) -> None:
    import numpy as np
    from scipy.io import wavfile

    fluidsynth = _review_dependency(
        "SIGILLUM_FLUIDSYNTH",
        "*-fluidsynth-*/bin/fluidsynth",
        "FluidSynth",
    )
    soundfont = _review_dependency(
        "SIGILLUM_REVIEW_SOUNDFONT",
        "*-Fluid-*/share/soundfonts/*.sf2",
        "FluidR3 General MIDI soundfont",
    )
    rendered = subprocess.run(
        (
            str(fluidsynth),
            "-ni",
            "-F",
            str(wav_path),
            "-r",
            "44100",
            str(soundfont),
            str(midi_path),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    if rendered.returncode != 0:
        raise SeamPlausibilityError(
            f"FluidSynth failed for {midi_path}: {rendered.stderr.strip()}"
        )
    sample_rate, audio = wavfile.read(wav_path)
    if sample_rate != 44_100:
        raise SeamPlausibilityError(
            f"FluidSynth rendered {midi_path} at {sample_rate} Hz"
        )
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0:
        raise SeamPlausibilityError(f"silent MIDI preview: {midi_path}")
    normalized = np.asarray(audio.astype(np.float64) / peak * 0.88 * 32767, dtype=np.int16)
    wavfile.write(wav_path, sample_rate, normalized)


def _review_dependency(
    environment_name: str,
    nix_pattern: str,
    label: str,
) -> Path:
    configured = os.environ.get(environment_name)
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.is_file():
            return path
        raise SeamPlausibilityError(
            f"{environment_name} does not name a file: {path}"
        )
    matches = sorted(Path("/nix/store").glob(nix_pattern))
    if matches:
        return matches[-1]
    raise SeamPlausibilityError(
        f"{label} is required for review audio. Install it with "
        "`nix build --no-link nixpkgs#fluidsynth nixpkgs#soundfont-fluid` "
        f"or set {environment_name}."
    )


def _contrast_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    review: list[dict[str, Any]] = []
    gross_controls: list[dict[str, Any]] = []
    for case in cases:
        candidates = {item["name"]: item for item in case["candidates"]}
        review.append(
            _contrast(
                case["case_id"],
                candidates["coherent_a"],
                candidates["hard_alternative"],
                "review",
            )
        )
        gross_controls.append(
            _contrast(
                case["case_id"],
                candidates["coherent_a"],
                candidates["discontinuous"],
                "gross_control",
            )
        )
    return {
        "review": {
            "comparison_count": len(review),
            "learned_prefers_authentic": sum(
                item["learned_preference"] == "preferred" for item in review
            ),
            "learned_prefers_alternative": sum(
                item["learned_preference"] == "other" for item in review
            ),
            "learned_ties": sum(
                item["learned_preference"] == "tie" for item in review
            ),
            "boundary_prefers_authentic": sum(
                item["boundary_preference"] == "preferred" for item in review
            ),
            "boundary_prefers_alternative": sum(
                item["boundary_preference"] == "other" for item in review
            ),
            "boundary_ties": sum(
                item["boundary_preference"] == "tie" for item in review
            ),
        },
        "gross_control": {
            "comparison_count": len(gross_controls),
            "learned_prefers_authentic": sum(
                item["learned_preference"] == "preferred"
                for item in gross_controls
            ),
            "boundary_prefers_authentic": sum(
                item["boundary_preference"] == "preferred"
                for item in gross_controls
            ),
        },
        "comparisons": review + gross_controls,
    }


def _contrast(
    case_id: str,
    preferred: dict[str, Any],
    other: dict[str, Any],
    difficulty: str,
) -> dict[str, Any]:
    learned_margin = preferred["learned_mean"] - other["learned_mean"]
    boundary_margin = preferred["boundary_mean"] - other["boundary_mean"]
    return {
        "case_id": case_id,
        "difficulty": difficulty,
        "preferred": preferred["name"],
        "other": other["name"],
        "learned_margin": learned_margin,
        "boundary_margin": boundary_margin,
        "learned_preference": _margin_preference(learned_margin),
        "boundary_preference": _margin_preference(boundary_margin),
    }


def _margin_preference(margin: float) -> str:
    if abs(margin) <= 1e-9:
        return "tie"
    return "preferred" if margin > 0 else "other"


def run(*, build_reviews: bool = False) -> dict[str, Any]:
    if not CHECKPOINT.is_file():
        raise SeamPlausibilityError(
            f"missing V4 checkpoint at {CHECKPOINT}; the frozen signal is required"
        )
    spec = StructuralContextSpec.load(CONTEXT_SPEC)
    scorer = StructuralSeamScorer.load(spec=spec, checkpoint_path=CHECKPOINT)
    if build_reviews:
        if REVIEW_ROOT.exists():
            shutil.rmtree(REVIEW_ROOT)
        (REVIEW_ROOT / "private").mkdir(parents=True)
        (REVIEW_ROOT / "bundles").mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="seam-plausibility-") as temp:
        directory = Path(temp)
        cases = [
            _score_case(
                directory=directory,
                case=case,
                scorer=scorer,
                review_root=REVIEW_ROOT if build_reviews else None,
            )
            for case in CASES
        ]
    report = {
        "experiment_id": "seam_plausibility_v1",
        "study_template": str(TEMPLATE.relative_to(ROOT)),
        "checkpoint_digest": _file_digest(CHECKPOINT),
        "context_spec_digest": spec.digest,
        "cases": cases,
        "summary": _contrast_summary(cases),
    }
    if build_reviews:
        report["review_manifest"] = _write_review_manifest(cases)
    return report


def _file_digest(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


# The generic sigillum-review-ui manifest schema (see tools/sigillum-review-ui).
# Seam plausibility is one producer of this cadence-agnostic contract.
REVIEW_OPTIONS = (
    {"id": "a", "label": "A is the more coherent return"},
    {"id": "b", "label": "B is the more coherent return"},
    {"id": "same", "label": "Same / cannot tell"},
)
REVIEW_QUESTION = "Which continuation is the more coherent return of the opening?"


def _review_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for case in cases:
        review = case["review"]
        items.append(
            {
                "id": review["review_id"],
                "title": review["title"],
                "question": REVIEW_QUESTION,
                "tags": ["seam", case["case_id"]],
                "variants": [
                    {
                        "id": label,
                        "label": label,
                        "audio": review["audio"][label],
                        "links": [
                            {"label": "MIDI", "path": review["midi"][label]},
                            {"label": "MusicXML", "path": review["musicxml"][label]},
                        ],
                    }
                    for label in ("A", "B")
                ],
            }
        )
    return {
        "schema_version": 1,
        "cadence_id": "seam_plausibility_v1",
        "title": "Structural seam plausibility",
        "description": (
            "Each item is a real Movement IV opening with two blinded "
            "continuations produced by the Partitura workflow. Choose the more "
            "coherent return, or Same."
        ),
        "response": {"kind": "single_choice", "options": list(REVIEW_OPTIONS)},
        "items": items,
    }


def _write_review_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = _review_manifest(cases)
    manifest_path = REVIEW_ROOT / "review-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "item_count": len(manifest["items"]),
        "cadence_id": manifest["cadence_id"],
        "manifest": str(manifest_path.relative_to(ROOT)),
        "media_root": str(REVIEW_ROOT.relative_to(ROOT)),
        "results_file": str(
            (REVIEW_ROOT / "review-results.json").relative_to(ROOT)
        ),
        "private_reviews": str(
            (REVIEW_ROOT / "private" / "reviews.jsonl").relative_to(ROOT)
        ),
    }


def _print(report: dict[str, Any]) -> None:
    print(f"checkpoint: {report['checkpoint_digest']}")
    for case in report["cases"]:
        print(f"\n{case['case_id']}: exported {case['exported_count']}/4")
        for candidate in case["candidates"]:
            print(
                f"  {candidate['name']:<15} {candidate['label']:<14} "
                f"learned={candidate['learned_mean']:+.6f} "
                f"boundary={candidate['boundary_mean']:+.6f} "
                f"residual={candidate['residual_mean']:+.6f}"
            )
        print(f"  learned:  {' > '.join(case['learned_ranking'])}")
        print(f"  boundary: {' > '.join(case['boundary_ranking'])}")
    review = report["summary"]["review"]
    gross = report["summary"]["gross_control"]
    print("\nreview contrasts (authentic vs plausible alternative)")
    print(
        "  learned: "
        f"{review['learned_prefers_authentic']} authentic / "
        f"{review['learned_prefers_alternative']} alternative / "
        f"{review['learned_ties']} ties"
    )
    print(
        "  boundary: "
        f"{review['boundary_prefers_authentic']} authentic / "
        f"{review['boundary_prefers_alternative']} alternative / "
        f"{review['boundary_ties']} ties"
    )
    print(
        "gross controls (authentic vs discontinuity): "
        f"learned {gross['learned_prefers_authentic']}/"
        f"{gross['comparison_count']}, boundary "
        f"{gross['boundary_prefers_authentic']}/{gross['comparison_count']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the machine-readable report under ignored outputs/",
    )
    arguments = parser.parse_args()
    report = run(build_reviews=arguments.write)
    _print(report)
    if arguments.write:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_ROOT / "ranking.json"
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
