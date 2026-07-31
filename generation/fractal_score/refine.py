"""Recursive refinement inference, cycle consistency, and honest evaluation."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import math

import numpy as np
import torch

from generation.fractal_score.baselines import Baseline
from generation.fractal_score.baselines import BigramLM
from generation.fractal_score.dataset import Window, refinement_arrays
from generation.fractal_score.ladder import (
    RefinementSchedule,
    refinement_positions,
    revealed_positions,
)
from generation.fractal_score.model import RefinementOperator
from generation.fractal_score.vocab import HarmonyVocab


@torch.no_grad()
def _predict(
    model: RefinementOperator,
    inputs: np.ndarray,
    level_index: int,
    key_ids: np.ndarray,
    pad_id: int,
    device: torch.device,
    forbid_ids: Sequence[int] | None = None,
) -> np.ndarray:
    model.eval()
    input_tensor = torch.from_numpy(inputs).to(device)
    level_tensor = torch.full(
        (inputs.shape[0],), level_index, dtype=torch.long, device=device
    )
    key_tensor = torch.from_numpy(key_ids).to(device)
    pad_mask = input_tensor == pad_id
    logits = model(input_tensor, level_tensor, key_tensor, pad_mask=pad_mask)
    if forbid_ids:
        # Control tokens (<pad>/<mask> and, for materialization, <none>/<rare>)
        # are never valid engraved content; block them before the argmax so a
        # refined slot always carries a real chord function.
        logits[:, :, list(forbid_ids)] = float("-inf")
    return logits.argmax(dim=-1).cpu().numpy()


def recursive_refine(
    model: RefinementOperator,
    vocab: HarmonyVocab,
    parent_ids: np.ndarray,
    key_id: int,
    schedule: RefinementSchedule,
    device: torch.device | None = None,
    forbid_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Engrave a coarse pillar grid into a full progression, one stride at a time.

    ``parent_ids`` carries true chords at multiples of the coarsest stride; other
    slots are ignored. Known slots are held fixed at every step (the fractal keeps
    the coarse skeleton), so parent preservation is exact by construction.

    ``forbid_ids`` are vocabulary ids blocked from the argmax at every step. It
    defaults to ``<pad>``/``<mask>`` (never valid content); callers materializing
    a human-facing progression also block ``<none>``/``<rare>``.
    """

    device = device or torch.device("cpu")
    if forbid_ids is None:
        forbid_ids = (vocab.pad_id, vocab.mask_id)
    length = int(parent_ids.shape[0])
    known = np.zeros((length,), dtype=bool)
    current = np.full((length,), vocab.mask_id, dtype=np.int64)
    for position in revealed_positions(length, schedule.coarsest):
        current[position] = parent_ids[position]
        known[position] = True
    key_ids = np.array([key_id], dtype=np.int64)
    for parent_stride, child_stride in schedule.steps():
        inputs = np.where(known, current, vocab.mask_id).astype(np.int64)[None, :]
        predicted = _predict(
            model,
            inputs,
            schedule.level_index(child_stride),
            key_ids,
            vocab.pad_id,
            device,
            forbid_ids=forbid_ids,
        )[0]
        for position in refinement_positions(length, parent_stride, child_stride):
            current[position] = predicted[position]
            known[position] = True
    return current


@dataclass(frozen=True)
class StepMetric:
    child_stride: int
    positions: int
    learned_accuracy: float
    baseline_accuracy: dict[str, float]


def _batch_ids(windows: Sequence[Window]) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.stack([window.token_ids for window in windows])
    key_ids = np.array([window.key_id for window in windows], dtype=np.int64)
    return token_ids, key_ids


def evaluate_steps(
    model: RefinementOperator,
    vocab: HarmonyVocab,
    windows: Sequence[Window],
    schedule: RefinementSchedule,
    baselines: Sequence[Baseline],
    device: torch.device | None = None,
    batch_size: int = 64,
) -> list[StepMetric]:
    """Per-step accuracy: given the true parent grid, predict the child slots."""

    device = device or torch.device("cpu")
    token_ids, key_ids = _batch_ids(windows)
    metrics: list[StepMetric] = []
    for parent_stride, child_stride in schedule.steps():
        inputs = np.stack(
            [
                refinement_arrays(
                    row,
                    parent_stride=parent_stride,
                    child_stride=child_stride,
                    mask_id=vocab.mask_id,
                    pad_id=vocab.pad_id,
                )[0]
                for row in token_ids
            ]
        )
        loss_masks = np.stack(
            [
                refinement_arrays(
                    row,
                    parent_stride=parent_stride,
                    child_stride=child_stride,
                    mask_id=vocab.mask_id,
                    pad_id=vocab.pad_id,
                )[2]
                for row in token_ids
            ]
        )
        predictions = np.empty_like(token_ids)
        for start in range(0, inputs.shape[0], batch_size):
            chunk = inputs[start : start + batch_size]
            keys = key_ids[start : start + batch_size]
            predictions[start : start + batch_size] = _predict(
                model,
                chunk,
                schedule.level_index(child_stride),
                keys,
                vocab.pad_id,
                device,
                forbid_ids=(vocab.pad_id, vocab.mask_id),
            )
        total = int(loss_masks.sum())
        learned = float((predictions[loss_masks] == token_ids[loss_masks]).mean()) if total else 0.0
        baseline_accuracy: dict[str, float] = {}
        for baseline in baselines:
            filled = np.stack(
                [
                    baseline.fill(row, parent_stride=parent_stride, child_stride=child_stride)
                    for row in token_ids
                ]
            )
            baseline_accuracy[baseline.name] = (
                float((filled[loss_masks] == token_ids[loss_masks]).mean()) if total else 0.0
            )
        metrics.append(
            StepMetric(
                child_stride=child_stride,
                positions=total,
                learned_accuracy=learned,
                baseline_accuracy=baseline_accuracy,
            )
        )
    return metrics


@dataclass(frozen=True)
class RecursiveMetric:
    positions: int
    learned_accuracy: float
    baseline_accuracy: dict[str, float]
    parent_preserved: bool


def evaluate_recursive(
    model: RefinementOperator,
    vocab: HarmonyVocab,
    windows: Sequence[Window],
    schedule: RefinementSchedule,
    baselines: Sequence[Baseline],
    device: torch.device | None = None,
) -> RecursiveMetric:
    """From coarse pillars only, recursively refine and score every filled slot."""

    device = device or torch.device("cpu")
    correct = 0
    total = 0
    parent_preserved = True
    baseline_correct = {baseline.name: 0 for baseline in baselines}
    for window in windows:
        length = len(window)
        truth = window.token_ids
        filled_positions = [
            index
            for index in range(length)
            if index % schedule.coarsest != 0 and truth[index] != vocab.pad_id
        ]
        if not filled_positions:
            continue
        generated = recursive_refine(
            model, vocab, truth, window.key_id, schedule, device=device
        )
        for position in revealed_positions(length, schedule.coarsest):
            if truth[position] != vocab.pad_id and generated[position] != truth[position]:
                parent_preserved = False
        for position in filled_positions:
            total += 1
            if generated[position] == truth[position]:
                correct += 1
        for baseline in baselines:
            chained = _baseline_recursive(baseline, truth, schedule, vocab)
            for position in filled_positions:
                if chained[position] == truth[position]:
                    baseline_correct[baseline.name] += 1
    learned = correct / total if total else 0.0
    baseline_accuracy = {
        name: value / total if total else 0.0 for name, value in baseline_correct.items()
    }
    return RecursiveMetric(
        positions=total,
        learned_accuracy=learned,
        baseline_accuracy=baseline_accuracy,
        parent_preserved=parent_preserved,
    )


def _baseline_recursive(
    baseline: Baseline,
    truth: np.ndarray,
    schedule: RefinementSchedule,
    vocab: HarmonyVocab,
) -> np.ndarray:
    length = int(truth.shape[0])
    current = np.full((length,), vocab.mask_id, dtype=np.int64)
    for position in revealed_positions(length, schedule.coarsest):
        current[position] = truth[position]
    for parent_stride, child_stride in schedule.steps():
        filled = baseline.fill(
            current, parent_stride=parent_stride, child_stride=child_stride
        )
        for position in refinement_positions(length, parent_stride, child_stride):
            current[position] = filled[position]
    return current


@dataclass(frozen=True)
class GenerativeMetric:
    """Fair likelihood comparison for a generative refiner.

    Exact-match rewards static copying because functional harmony persists
    bar-to-bar. Held-out negative log-likelihood instead asks whether the learned
    operator assigns more probability to the true finer chord than a bigram
    language model that only sees the previous bar.
    """

    positions: int
    learned_nll: float
    bigram_nll: float


@torch.no_grad()
def evaluate_generative(
    model: RefinementOperator,
    vocab: HarmonyVocab,
    windows: Sequence[Window],
    schedule: RefinementSchedule,
    bigram_lm: BigramLM,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> GenerativeMetric:
    device = device or torch.device("cpu")
    model.eval()
    token_ids, key_ids = _batch_ids(windows)
    total = 0
    learned_sum = 0.0
    bigram_sum = 0.0
    for parent_stride, child_stride in schedule.steps():
        rows_in, rows_mask = [], []
        for row in token_ids:
            row_in, _, row_mask = refinement_arrays(
                row,
                parent_stride=parent_stride,
                child_stride=child_stride,
                mask_id=vocab.mask_id,
                pad_id=vocab.pad_id,
            )
            rows_in.append(row_in)
            rows_mask.append(row_mask)
        inputs = np.stack(rows_in)
        masks = np.stack(rows_mask)
        level = schedule.level_index(child_stride)
        for start in range(0, inputs.shape[0], batch_size):
            chunk = torch.from_numpy(inputs[start : start + batch_size]).to(device)
            keys = torch.from_numpy(key_ids[start : start + batch_size]).to(device)
            levels = torch.full((chunk.size(0),), level, dtype=torch.long, device=device)
            pad_mask = chunk == vocab.pad_id
            logits = model(chunk, levels, keys, pad_mask=pad_mask)
            log_probs = torch.log_softmax(logits, dim=-1)
            targets = token_ids[start : start + batch_size]
            mask_chunk = masks[start : start + batch_size]
            for b in range(chunk.size(0)):
                for position in np.nonzero(mask_chunk[b])[0]:
                    target = int(targets[b, position])
                    learned_sum += -float(log_probs[b, position, target].item())
                    previous = int(targets[b, position - 1]) if position > 0 else target
                    bigram_sum += -bigram_lm.log_prob(previous, target)
                    total += 1
    return GenerativeMetric(
        positions=total,
        learned_nll=learned_sum / total if total else 0.0,
        bigram_nll=bigram_sum / total if total else 0.0,
    )


@dataclass(frozen=True)
class RealismMetric:
    """How closely a refinement's harmonic motion matches the authentic surface.

    ``repeat_rate`` is the fraction of adjacent bars that keep the same chord.
    Copying a coarse skeleton inflates repeats; a good refiner should land near
    the authentic repeat rate and match its chord-change distribution (low
    Jensen-Shannon divergence to the authentic bigram distribution).
    """

    authentic_repeat_rate: float
    learned_repeat_rate: float
    baseline_repeat_rate: dict[str, float]
    learned_js: float
    baseline_js: dict[str, float]


def _repeat_rate(sequence: np.ndarray) -> float:
    if sequence.shape[0] < 2:
        return 0.0
    return float(np.mean(sequence[1:] == sequence[:-1]))


def _bigram_distribution(sequences: Sequence[np.ndarray]) -> dict[tuple[int, int], float]:
    counts: dict[tuple[int, int], float] = {}
    total = 0.0
    for sequence in sequences:
        for previous, current in zip(sequence[:-1], sequence[1:]):
            key = (int(previous), int(current))
            counts[key] = counts.get(key, 0.0) + 1.0
            total += 1.0
    if total == 0.0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _jensen_shannon(
    left: dict[tuple[int, int], float], right: dict[tuple[int, int], float]
) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    js = 0.0
    for key in keys:
        p = left.get(key, 0.0)
        q = right.get(key, 0.0)
        m = 0.5 * (p + q)
        if p > 0.0:
            js += 0.5 * p * math.log(p / m)
        if q > 0.0:
            js += 0.5 * q * math.log(q / m)
    return js / math.log(2.0)


def evaluate_realism(
    model: RefinementOperator,
    vocab: HarmonyVocab,
    windows: Sequence[Window],
    schedule: RefinementSchedule,
    baselines: Sequence[Baseline],
    device: torch.device | None = None,
) -> RealismMetric:
    device = device or torch.device("cpu")
    authentic: list[np.ndarray] = []
    learned: list[np.ndarray] = []
    baseline_sequences: dict[str, list[np.ndarray]] = {b.name: [] for b in baselines}
    for window in windows:
        truth = window.token_ids
        length = len(window)
        valid = truth != vocab.pad_id
        if valid.sum() < 2:
            continue
        authentic.append(truth[valid])
        generated = recursive_refine(
            model, vocab, truth, window.key_id, schedule, device=device
        )
        learned.append(generated[valid])
        for baseline in baselines:
            chained = _baseline_recursive(baseline, truth, schedule, vocab)
            baseline_sequences[baseline.name].append(chained[valid])
    authentic_bigram = _bigram_distribution(authentic)
    learned_bigram = _bigram_distribution(learned)
    return RealismMetric(
        authentic_repeat_rate=float(np.mean([_repeat_rate(seq) for seq in authentic])),
        learned_repeat_rate=float(np.mean([_repeat_rate(seq) for seq in learned])),
        baseline_repeat_rate={
            name: float(np.mean([_repeat_rate(seq) for seq in seqs]))
            for name, seqs in baseline_sequences.items()
        },
        learned_js=_jensen_shannon(learned_bigram, authentic_bigram),
        baseline_js={
            name: _jensen_shannon(_bigram_distribution(seqs), authentic_bigram)
            for name, seqs in baseline_sequences.items()
        },
    )
