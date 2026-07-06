"""Compact human-readable summary of a candidate spread.

Renders each candidate's realized notes plus its actual free-region contour peak, surprise
peak, repetition echoes, and pin satisfaction. Ordering reflects dissimilarity, not merit
(spec Output section). Decision D5 (``docs/architecture/17_theme_generator.md``) folds
kernel density / free-space figures in here (plan M5). This module carries the original
prototype's reporting layer forward (plan M0).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from generation.theme_gen._common import MelodyItem
from generation.theme_gen.density import kernel_density

if TYPE_CHECKING:
    from generation.theme_gen.engine import GenerationTrace, ThemeCandidate
    from generation.theme_gen.kerneldsl import ThemeKernel


def render_run_log(trace: GenerationTrace, candidates: Sequence[ThemeCandidate]) -> str:
    """The generation run-log: how the pool was filled and which candidates survived.

    Surfaces the Stage-1 degradation the pipeline otherwise hides — attempts abandoned
    (feasibility/backtracking exhausted), duplicates dropped, pins failed — and, per selected
    candidate, where in the kept pool it landed and how many attempts were dropped just before
    it. Ordering of the selected list is spread order, not a quality ranking.
    """
    outcomes = {"kept": 0, "abandoned": 0, "duplicate": 0, "pin_fail": 0}
    for attempt in trace.attempts:
        outcomes[attempt.outcome] = outcomes.get(attempt.outcome, 0) + 1
    kept = [a for a in trace.attempts if a.outcome == "kept"]

    # Attempts dropped (any non-kept outcome) since the previous kept candidate, per pool index.
    drops_before: dict[int, int] = {}
    run = 0
    for attempt in trace.attempts:
        if attempt.outcome == "kept":
            drops_before[attempt.pool_index] = run
            run = 0
        else:
            run += 1

    kept_backtracks = [a.backtracks for a in kept]
    total_bt = sum(a.backtracks for a in trace.attempts)
    avg_bt = (sum(kept_backtracks) / len(kept_backtracks)) if kept_backtracks else 0.0
    max_bt = max((a.backtracks for a in trace.attempts), default=0)

    soft_kept = sum(1 for a in kept if a.soft_pin_fails)
    lines = [
        f"attempts: {len(trace.attempts)} total — {outcomes['kept']} kept · "
        f"{outcomes['abandoned']} abandoned · {outcomes['duplicate']} duplicate · "
        f"{outcomes['pin_fail']} pin-fail [hard pins] (pool target {trace.pool_target})",
        f"backtracks: {total_bt} total, {avg_bt:.1f} avg/kept, {max_bt} max",
        f"soft-pin misses: {soft_kept} of {len(kept)} kept miss a harmonic/cadence "
        f"implication (guided toward, not dropped)",
        f"spread: selected {len(trace.selected)} of {len(trace.kept_after_cut)} kept "
        f"(after conformance keep-cut)",
        "",
        "selected (spread order):",
    ]
    for candidate in candidates:
        dropped = drops_before.get(candidate.pool_index, 0)
        tail = f" — dropped {dropped} before it" if dropped else ""
        soft = f"  soft {candidate.soft_pin_fails}" if candidate.soft_pin_fails else ""
        lines.append(
            f"- pool #{candidate.pool_index}  conf {candidate.conformance:.3f}  "
            f"bt {candidate.backtracks}{soft}{tail}  | {_format_items(candidate.items)}"
        )
    return "\n".join(lines)


def render_kernel_density(kernel: ThemeKernel, batch_size: int | None = None) -> str:
    """Density / free-space / expected-spread block for a kernel (D5).

    The tighter the kernel, the narrower the achievable spread — surfaced, not hidden.
    """
    d = kernel_density(kernel)
    realizations = f">={d.feasible_realizations}" if d.realizations_saturated else str(d.feasible_realizations)
    lines = [
        f"Density: {d.density * 100:.0f}% pinned ({d.pinned_ql:g}/{d.total_ql:g} ql); free space {d.free_space_fraction * 100:.0f}%",
        f"Expected spread: {d.expected_spread_bits:.1f} bits ({realizations} feasible realizations)",
    ]
    if batch_size is not None and not d.realizations_saturated and d.feasible_realizations < batch_size:
        lines.append(
            f"NOTE: kernel admits only {d.feasible_realizations} realizations (< batch {batch_size}); expect near-duplicates."
        )
    return "\n".join(lines)


def render_candidate_report(
    candidates: Sequence[ThemeCandidate],
    kernel: ThemeKernel,
    trace: "GenerationTrace | None" = None,
) -> str:
    lines = [
        "# Kernel Candidate Spread",
        "",
        f"Frame: {kernel.frame.bars} bars, {kernel.frame.meter}, {kernel.frame.key}, role={kernel.frame.role}",
        f"Pitch/rhythm pins: {len(kernel.pitch_rhythm_pins)}; harmonic pins: {len(kernel.harmonic_pins)}; structural pins: {len(kernel.structural_pins)}",
        render_kernel_density(kernel, batch_size=len(candidates) or None),
        "",
        "Ordering is spread-selection order, not a quality ranking.",
        "",
    ]
    if trace is not None:
        lines += ["## Generation run-log", "", render_run_log(trace, candidates), ""]
    for index, candidate in enumerate(candidates, 1):
        contour_peak = _peak_bar(candidate.features.contour)
        surprise_peak = _peak_bar(candidate.features.surprise)
        echoes = _echo_pairs(candidate.features.similarity)
        lines.extend(
            [
                f"## Candidate {index}",
                f"- conformance distance: {candidate.conformance:.3f}",
                f"- rolled target: contour={candidate.target.contour_kind}, surprise_peak={candidate.target.surprise_peak_bar}",
                f"- actual free contour peak: b{contour_peak}; actual surprise peak: b{surprise_peak}",
                f"- free-bar echoes: {', '.join(echoes) if echoes else 'none above threshold'}",
                f"- pins: {'; '.join(candidate.pin_report)}",
                f"- items: {_format_items(candidate.items)}",
                "",
            ]
        )
    return "\n".join(lines)


def _peak_bar(values: Sequence[float]) -> int:
    if not values:
        return 0
    return max(range(len(values)), key=lambda i: values[i]) + 1


def _echo_pairs(matrix: Sequence[Sequence[float]], threshold: float = 0.64) -> list[str]:
    pairs = []
    for i, row in enumerate(matrix):
        for j in range(i + 1, len(row)):
            if row[j] >= threshold:
                pairs.append(f"b{i + 1}=b{j + 1}")
    return pairs[:5]


def _format_items(items: Sequence[MelodyItem]) -> str:
    return " ".join(f"{item[0]}:{float(item[1]):g}" for item in items)
