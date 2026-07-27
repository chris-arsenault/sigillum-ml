"""Aggregation and held-out reporting for whole-score benchmark evidence."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from generation.composition.benchmark import (
    HUMAN_CRITERIA,
    BenchmarkCell,
    BenchmarkError,
    BenchmarkManifest,
    _array,
    _freeze,
    _mapping,
    _text,
    _validate_digest,
)
from generation.composition.evaluation_run import EvaluationRun

EVALUATION_REVIEW_SCHEMA_VERSION = 1
EVALUATION_PREFERENCE_SCHEMA_VERSION = 1
_OUTCOMES = {"a", "b", "tie", "abstain"}


def _jsonl(path: str | Path, label: str) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    source_path = Path(path)
    if not source_path.exists():
        return ()
    with source_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                records.append(_mapping(json.loads(line), label))
            except (json.JSONDecodeError, BenchmarkError) as error:
                raise BenchmarkError(
                    f"invalid {label} at line {line_number}: {error}"
                ) from error
    return tuple(records)


@dataclass(frozen=True)
class ScoreReviewRecord:
    review_id: str
    benchmark_id: str
    case_id: str
    criterion: str
    subjects: Mapping[str, str]
    source_digests: Mapping[str, str]
    snapshot_digests: Mapping[str, str]
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScoreReviewRecord:
        if value.get("schema_version") != EVALUATION_REVIEW_SCHEMA_VERSION:
            raise BenchmarkError("unsupported score evaluation review schema")
        if value.get("kind") != "score_evaluation_review":
            raise BenchmarkError("invalid score evaluation review kind")
        if value.get("blind") is not True:
            raise BenchmarkError("score evaluation review must be blinded")
        criterion = _text(value.get("criterion"), "review criterion")
        if criterion not in HUMAN_CRITERIA:
            raise BenchmarkError(f"unsupported review criterion {criterion!r}")
        subjects = {}
        source_digests = {}
        snapshot_digests = {}
        for item in _array(value.get("subjects"), "review subjects"):
            subject = _mapping(item, "review subject")
            label = _text(subject.get("label"), "review label")
            run_id = _text(subject.get("run_id"), "review run_id")
            subjects[label] = run_id
            source_digests[label] = _validate_digest(
                subject.get("source_digest"), "review source_digest"
            )
            snapshot_digests[label] = _validate_digest(
                subject.get("snapshot_digest"), "review snapshot_digest"
            )
        if set(subjects) != {"A", "B"} or len(set(subjects.values())) != 2:
            raise BenchmarkError("review requires distinct A and B runs")
        return cls(
            review_id=_text(value.get("review_id"), "review_id"),
            benchmark_id=_text(value.get("benchmark_id"), "review benchmark_id"),
            case_id=_text(value.get("case_id"), "review case_id"),
            criterion=criterion,
            subjects=MappingProxyType(subjects),
            source_digests=MappingProxyType(source_digests),
            snapshot_digests=MappingProxyType(snapshot_digests),
            raw=_freeze(value),
        )


@dataclass(frozen=True)
class ScorePreferenceRecord:
    preference_id: str
    review_id: str
    benchmark_id: str
    case_id: str
    criterion: str
    outcome: str
    preferred_run_id: str | None
    other_run_id: str | None
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ScorePreferenceRecord:
        if value.get("schema_version") != EVALUATION_PREFERENCE_SCHEMA_VERSION:
            raise BenchmarkError("unsupported score evaluation preference schema")
        if value.get("kind") != "score_evaluation_preference":
            raise BenchmarkError("invalid score evaluation preference kind")
        if value.get("blind") is not True:
            raise BenchmarkError("score evaluation preference must be blinded")
        if value.get("purpose") != "held_out_evaluation":
            raise BenchmarkError("score evaluation preferences must remain held out")
        outcome = _text(value.get("outcome"), "preference outcome")
        if outcome not in _OUTCOMES:
            raise BenchmarkError(f"unsupported preference outcome {outcome!r}")
        preferred = value.get("preferred_run_id")
        other = value.get("other_run_id")
        if outcome in {"a", "b"}:
            preferred = _text(preferred, "preferred_run_id")
            other = _text(other, "other_run_id")
            if preferred == other:
                raise BenchmarkError("preference run ids must be distinct")
        elif preferred is not None or other is not None:
            raise BenchmarkError("tie and abstain may not resolve run ids")
        criterion = _text(value.get("criterion"), "preference criterion")
        if criterion not in HUMAN_CRITERIA:
            raise BenchmarkError(f"unsupported preference criterion {criterion!r}")
        return cls(
            preference_id=_text(value.get("preference_id"), "preference_id"),
            review_id=_text(value.get("review_id"), "preference review_id"),
            benchmark_id=_text(value.get("benchmark_id"), "preference benchmark_id"),
            case_id=_text(value.get("case_id"), "preference case_id"),
            criterion=criterion,
            outcome=outcome,
            preferred_run_id=preferred,
            other_run_id=other,
            raw=_freeze(value),
        )


@dataclass(frozen=True)
class ComparisonCell:
    case_id: str
    seed: int
    criterion: str
    left_system: tuple[str, str]
    right_system: tuple[str, str]

    @classmethod
    def create(
        cls,
        *,
        case_id: str,
        seed: int,
        criterion: str,
        first: tuple[str, str],
        second: tuple[str, str],
    ) -> ComparisonCell:
        left, right = sorted((first, second))
        return cls(case_id, seed, criterion, left, right)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "seed": self.seed,
            "criterion": self.criterion,
            "left": {
                "strategy_id": self.left_system[0],
                "ablation_id": self.left_system[1],
            },
            "right": {
                "strategy_id": self.right_system[0],
                "ablation_id": self.right_system[1],
            },
        }


class EvaluationLab:
    """A validated, immutable join over runs and held-out human evidence."""

    def __init__(
        self,
        manifest: BenchmarkManifest,
        runs: Iterable[EvaluationRun],
        reviews: Iterable[ScoreReviewRecord] = (),
        preferences: Iterable[ScorePreferenceRecord] = (),
    ) -> None:
        self.manifest = manifest
        self.runs = tuple(runs)
        self.reviews = tuple(reviews)
        self.preferences = tuple(preferences)
        self._runs_by_id = self._unique(self.runs, "run_id", "run")
        self._reviews_by_id = self._unique(self.reviews, "review_id", "review")
        self._unique(self.preferences, "preference_id", "preference")
        self._unique(self.preferences, "review_id", "rated review")
        self._validate_runs()
        self._validate_reviews()
        self._validate_preferences()

    @classmethod
    def from_jsonl(
        cls,
        manifest: BenchmarkManifest,
        run_path: str | Path,
        review_path: str | Path | None = None,
        preference_path: str | Path | None = None,
    ) -> EvaluationLab:
        runs = tuple(EvaluationRun.from_dict(item) for item in _jsonl(run_path, "run"))
        reviews = (
            tuple(
                ScoreReviewRecord.from_dict(item)
                for item in _jsonl(review_path, "review")
            )
            if review_path
            else ()
        )
        preferences = (
            tuple(
                ScorePreferenceRecord.from_dict(item)
                for item in _jsonl(preference_path, "preference")
            )
            if preference_path
            else ()
        )
        return cls(manifest, runs, reviews, preferences)

    def report(self) -> EvaluationReport:
        return EvaluationReport(self)

    def expected_comparisons(self) -> tuple[ComparisonCell, ...]:
        control_systems = [
            (strategy.strategy_id, "control") for strategy in self.manifest.strategies
        ]
        pairs = list(combinations(control_systems, 2))
        pairs.extend(
            (
                (strategy_id, "control"),
                (strategy_id, ablation.ablation_id),
            )
            for ablation in self.manifest.ablations
            for strategy_id in ablation.strategy_ids
        )
        return tuple(
            ComparisonCell.create(
                case_id=case.case_id,
                seed=seed,
                criterion=criterion,
                first=first,
                second=second,
            )
            for case in self.manifest.cases
            for seed in self.manifest.seeds
            for criterion in self.manifest.human_criteria
            for first, second in pairs
        )

    def observed_comparisons(self) -> tuple[ComparisonCell, ...]:
        rated = {item.review_id for item in self.preferences}
        return tuple(
            self._comparison_for(review)
            for review in self.reviews
            if review.review_id in rated
        )

    def ready_review_assignments(self) -> tuple[dict[str, Any], ...]:
        runs_by_cell = {run.cell.key(): run for run in self.runs}
        observed = set(self.observed_comparisons())
        assignments = []
        for comparison in self.expected_comparisons():
            if comparison in observed:
                continue
            left = runs_by_cell.get(
                (
                    comparison.case_id,
                    comparison.left_system[0],
                    comparison.left_system[1],
                    comparison.seed,
                )
            )
            right = runs_by_cell.get(
                (
                    comparison.case_id,
                    comparison.right_system[0],
                    comparison.right_system[1],
                    comparison.seed,
                )
            )
            if left and right:
                assignments.append(
                    {
                        "benchmark_id": self.manifest.benchmark_id,
                        "case_id": comparison.case_id,
                        "seed": comparison.seed,
                        "criterion": comparison.criterion,
                        "left_run_id": left.run_id,
                        "right_run_id": right.run_id,
                    }
                )
        return tuple(assignments)

    def _validate_runs(self) -> None:
        expected = {cell.key() for cell in self.manifest.expected_cells()}
        seen = set()
        for run in self.runs:
            if (
                run.manifest_digest != self.manifest.manifest_digest
                or run.benchmark_id != self.manifest.benchmark_id
            ):
                raise BenchmarkError(f"run {run.run_id} targets another benchmark")
            key = run.cell.key()
            if key not in expected:
                raise BenchmarkError(f"run {run.run_id} targets an unexpected cell")
            if key in seen:
                raise BenchmarkError(f"benchmark cell is repeated: {key}")
            seen.add(key)

    def _validate_reviews(self) -> None:
        expected = set(self.expected_comparisons())
        for review in self.reviews:
            if review.benchmark_id != self.manifest.benchmark_id:
                raise BenchmarkError(
                    f"review {review.review_id} targets another benchmark"
                )
            comparison = self._comparison_for(review)
            self._validate_review_sources(review)
            if comparison not in expected:
                raise BenchmarkError(
                    f"review {review.review_id} is outside the frozen design"
                )

    def _validate_review_sources(self, review: ScoreReviewRecord) -> None:
        for label, run_id in review.subjects.items():
            run = self._runs_by_id[run_id]
            measurement = run.measurement
            if review.source_digests[label] != measurement["source_digest"]:
                raise BenchmarkError(
                    f"review {review.review_id} source differs from run {run_id}"
                )
            if review.snapshot_digests[label] != measurement.get("snapshot_digest"):
                raise BenchmarkError(
                    f"review {review.review_id} snapshot differs from run {run_id}"
                )

    def _validate_preferences(self) -> None:
        for preference in self.preferences:
            review = self._reviews_by_id.get(preference.review_id)
            if review is None:
                raise BenchmarkError(
                    f"preference {preference.preference_id} has no review"
                )
            if (
                preference.benchmark_id != review.benchmark_id
                or preference.case_id != review.case_id
                or preference.criterion != review.criterion
            ):
                raise BenchmarkError(
                    f"preference {preference.preference_id} disagrees with its review"
                )
            if preference.outcome not in {"a", "b"}:
                continue
            chosen = "A" if preference.outcome == "a" else "B"
            other = "B" if chosen == "A" else "A"
            if (
                preference.preferred_run_id != review.subjects[chosen]
                or preference.other_run_id != review.subjects[other]
            ):
                raise BenchmarkError(
                    f"preference {preference.preference_id} has the wrong blind mapping"
                )

    def _comparison_for(self, review: ScoreReviewRecord) -> ComparisonCell:
        first = self._runs_by_id.get(review.subjects["A"])
        second = self._runs_by_id.get(review.subjects["B"])
        if first is None or second is None:
            raise BenchmarkError(f"review {review.review_id} names an absent run")
        if (
            first.cell.case_id != second.cell.case_id
            or first.cell.case_id != review.case_id
        ):
            raise BenchmarkError(f"review {review.review_id} mixes benchmark cases")
        if first.cell.seed != second.cell.seed:
            raise BenchmarkError(f"review {review.review_id} mixes benchmark seeds")
        return ComparisonCell.create(
            case_id=review.case_id,
            seed=first.cell.seed,
            criterion=review.criterion,
            first=(first.cell.strategy_id, first.cell.ablation_id),
            second=(second.cell.strategy_id, second.cell.ablation_id),
        )

    @staticmethod
    def _unique(records: Iterable[Any], attribute: str, label: str) -> dict[str, Any]:
        indexed = {}
        for record in records:
            record_id = getattr(record, attribute)
            if record_id in indexed:
                raise BenchmarkError(f"duplicate {label} id {record_id}")
            indexed[record_id] = record
        return indexed


class EvaluationReport:
    """Plural diagnostics and preferences; intentionally no composite reward."""

    def __init__(self, lab: EvaluationLab) -> None:
        self.lab = lab

    def to_dict(self) -> dict[str, Any]:
        missing_runs = self._missing_runs()
        missing_comparisons = self._missing_comparisons()
        return {
            "schema_version": 1,
            "kind": "whole_score_evaluation_report",
            "benchmark_id": self.lab.manifest.benchmark_id,
            "manifest_digest": self.lab.manifest.manifest_digest,
            "status": (
                "complete"
                if not missing_runs and not missing_comparisons
                else "incomplete"
            ),
            "coverage": {
                "expected_runs": len(self.lab.manifest.expected_cells()),
                "observed_runs": len(self.lab.runs),
                "missing_runs": [cell.to_dict() for cell in missing_runs],
                "expected_human_comparisons": len(self.lab.expected_comparisons()),
                "observed_human_comparisons": len(set(self.lab.observed_comparisons())),
                "missing_human_comparisons": [
                    cell.to_dict() for cell in missing_comparisons
                ],
                "ready_review_assignments": list(self.lab.ready_review_assignments()),
            },
            "systems": self._system_reports(),
            "warnings": [
                "Partitura diagnostics are descriptive proxies, not musical-quality scores.",
                "Human preferences are held out and are never collapsed into a training reward here.",
                "No composite score or automatic winner is produced.",
            ],
        }

    def render_markdown(self) -> str:
        report = self.to_dict()
        coverage = report["coverage"]
        lines = [
            f"# Whole-score evaluation: {report['benchmark_id']}",
            "",
            f"Status: **{report['status']}**",
            "",
            f"Runs: {coverage['observed_runs']}/{coverage['expected_runs']}; "
            f"human comparisons: {coverage['observed_human_comparisons']}/"
            f"{coverage['expected_human_comparisons']}.",
            "",
            "| System | Runs | Valid | Bound | Identity | Seam silence | "
            "Accepted/candidate | Diversity |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for system in report["systems"]:
            diagnostics = system["diagnostics"]
            lines.append(
                f"| {system['strategy_id']} / {system['ablation_id']} "
                f"| {system['run_count']} "
                f"| {self._format(system['mechanical_valid_rate'])} "
                f"| {self._format(diagnostics['requirement_bound_ratio'])} "
                f"| {self._format(diagnostics['material_linked_phrase_ratio'])} "
                f"| {self._format(diagnostics['silent_boundary_ratio'])} "
                f"| {self._format(system['edit_efficiency']['accepted_per_candidate'])} "
                f"| {self._format(system['diversity']['mean_ngram_jaccard_distance'])} |"
            )
        lines.extend(["", "No composite score or automatic winner is produced."])
        return "\n".join(lines)

    def _system_reports(self) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[EvaluationRun]] = {}
        for run in self.lab.runs:
            key = (run.cell.strategy_id, run.cell.ablation_id)
            groups.setdefault(key, []).append(run)
        human = self._human_counts()
        return [
            self._system_report(key, runs, human.get(key, {}))
            for key, runs in sorted(groups.items())
        ]

    def _system_report(
        self,
        key: tuple[str, str],
        runs: list[EvaluationRun],
        human: Mapping[str, Mapping[str, int]],
    ) -> dict[str, Any]:
        valid = [run for run in runs if run.measurement["mechanical"]["valid"]]
        candidates = sum(run.effort.candidate_count for run in runs)
        accepted = sum(run.effort.accepted_edit_count for run in runs)
        return {
            "strategy_id": key[0],
            "ablation_id": key[1],
            "run_count": len(runs),
            "mechanical_valid_rate": round(len(valid) / len(runs), 6),
            "diagnostics": {
                "requirement_bound_ratio": self._mean_path(
                    valid, "diagnostics", "requirements", "bound_ratio"
                ),
                "material_linked_phrase_ratio": self._mean_path(
                    valid,
                    "diagnostics",
                    "identity",
                    "material_linked_phrase_ratio",
                ),
                "silent_boundary_ratio": self._boundary_ratio(valid, "silent"),
                "sustained_boundary_ratio": self._boundary_ratio(valid, "sustained"),
                "silence_ratio": self._mean_path(
                    valid, "diagnostics", "reserve", "silence_ratio"
                ),
            },
            "edit_efficiency": {
                "accepted_per_candidate": (
                    round(accepted / candidates, 6) if candidates else None
                ),
                "mechanically_valid_per_candidate": (
                    round(
                        sum(
                            run.effort.mechanically_valid_candidate_count
                            for run in runs
                        )
                        / candidates,
                        6,
                    )
                    if candidates
                    else None
                ),
                "mean_model_calls": self._mean(
                    run.effort.model_call_count for run in runs
                ),
                "mean_wall_seconds": self._mean(
                    run.effort.wall_seconds for run in runs
                ),
            },
            "diversity": self._diversity(valid),
            "human_preference": {
                criterion: dict(
                    human.get(
                        criterion,
                        {"wins": 0, "losses": 0, "ties": 0, "abstentions": 0},
                    )
                )
                for criterion in self.lab.manifest.human_criteria
            },
        }

    def _human_counts(self) -> dict[tuple[str, str], dict[str, dict[str, int]]]:
        counts: dict[tuple[str, str], dict[str, dict[str, int]]] = {}
        reviews = {item.review_id: item for item in self.lab.reviews}
        runs = {item.run_id: item for item in self.lab.runs}
        for preference in self.lab.preferences:
            review = reviews[preference.review_id]
            subjects = [runs[review.subjects[label]] for label in ("A", "B")]
            keys = [(run.cell.strategy_id, run.cell.ablation_id) for run in subjects]
            entries = [
                counts.setdefault(key, {}).setdefault(
                    preference.criterion,
                    {"wins": 0, "losses": 0, "ties": 0, "abstentions": 0},
                )
                for key in keys
            ]
            if preference.outcome == "tie":
                entries[0]["ties"] += 1
                entries[1]["ties"] += 1
            elif preference.outcome == "abstain":
                entries[0]["abstentions"] += 1
                entries[1]["abstentions"] += 1
            else:
                preferred = runs[preference.preferred_run_id]
                winner = (preferred.cell.strategy_id, preferred.cell.ablation_id)
                for key, entry in zip(keys, entries):
                    entry["wins" if key == winner else "losses"] += 1
        return counts

    def _missing_runs(self) -> tuple[BenchmarkCell, ...]:
        observed = {run.cell.key() for run in self.lab.runs}
        return tuple(
            cell
            for cell in self.lab.manifest.expected_cells()
            if cell.key() not in observed
        )

    def _missing_comparisons(self) -> tuple[ComparisonCell, ...]:
        observed = set(self.lab.observed_comparisons())
        return tuple(
            cell for cell in self.lab.expected_comparisons() if cell not in observed
        )

    @staticmethod
    def _mean(values: Iterable[float]) -> float | None:
        present = tuple(values)
        return round(statistics.fmean(present), 6) if present else None

    def _mean_path(self, runs: list[EvaluationRun], *path: str) -> float | None:
        values = []
        for run in runs:
            value: Any = run.measurement
            for key in path:
                value = value.get(key) if isinstance(value, Mapping) else None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
        return self._mean(values)

    def _boundary_ratio(self, runs: list[EvaluationRun], kind: str) -> float | None:
        ratios = []
        for run in runs:
            seams = run.measurement["diagnostics"]["seams"]
            boundaries = seams["boundary_count"]
            if boundaries:
                ratios.append(seams[f"{kind}_boundary_count"] / boundaries)
        return self._mean(ratios)

    def _diversity(self, runs: list[EvaluationRun]) -> dict[str, Any]:
        fingerprints = [
            set(run.measurement["fingerprints"]["event_ngrams"]) for run in runs
        ]
        score_ids = [run.measurement["fingerprints"]["score"] for run in runs]
        distances = []
        for left, right in combinations(fingerprints, 2):
            union = left | right
            distances.append(0.0 if not union else 1.0 - len(left & right) / len(union))
        return {
            "unique_score_ratio": (
                round(len(set(score_ids)) / len(score_ids), 6) if score_ids else None
            ),
            "mean_ngram_jaccard_distance": self._mean(distances),
        }

    @staticmethod
    def _format(value: Any) -> str:
        return "—" if value is None else f"{value:.3f}"
