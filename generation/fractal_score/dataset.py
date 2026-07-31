"""Windowing and masked-refinement example construction (framework-only, torch-free)."""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from generation.fractal_score.harmony import HarmonyProgression
from generation.fractal_score.ladder import refinement_positions, revealed_positions
from generation.fractal_score.vocab import HarmonyVocab


class DatasetError(ValueError):
    """Raised when windows or refinement examples cannot be constructed."""


@dataclass(frozen=True)
class Window:
    """A fixed-length slice of one score's per-bar progression."""

    token_ids: np.ndarray
    key_id: int
    score_id: str
    lineage_id: str
    split: str
    start_index: int

    def __len__(self) -> int:
        return int(self.token_ids.shape[0])


def extract_windows(
    progressions: Iterable[HarmonyProgression],
    vocab: HarmonyVocab,
    *,
    window: int,
    stride: int,
) -> list[Window]:
    if window < 2 or stride < 1:
        raise DatasetError("window must be >= 2 and stride >= 1")
    windows: list[Window] = []
    for progression in progressions:
        ids = np.array(
            [vocab.encode(token) for token in progression.tokens],
            dtype=np.int64,
        )
        key_id = vocab.encode_key(progression.home_key)
        length = ids.shape[0]
        if length < window:
            windows.append(
                _padded_window(ids, vocab, key_id, progression, window, 0)
            )
            continue
        for start in range(0, length - window + 1, stride):
            windows.append(
                Window(
                    token_ids=ids[start : start + window].copy(),
                    key_id=key_id,
                    score_id=progression.score_id,
                    lineage_id=progression.lineage_id,
                    split=progression.split,
                    start_index=progression.first_measure + start,
                )
            )
        tail_start = length - window
        if tail_start % stride != 0:
            windows.append(
                Window(
                    token_ids=ids[tail_start:].copy(),
                    key_id=key_id,
                    score_id=progression.score_id,
                    lineage_id=progression.lineage_id,
                    split=progression.split,
                    start_index=progression.first_measure + tail_start,
                )
            )
    if not windows:
        raise DatasetError("no windows produced from the given progressions")
    return windows


def _padded_window(
    ids: np.ndarray,
    vocab: HarmonyVocab,
    key_id: int,
    progression: HarmonyProgression,
    window: int,
    start: int,
) -> Window:
    padded = np.full((window,), vocab.pad_id, dtype=np.int64)
    padded[: ids.shape[0]] = ids
    return Window(
        token_ids=padded,
        key_id=key_id,
        score_id=progression.score_id,
        lineage_id=progression.lineage_id,
        split=progression.split,
        start_index=progression.first_measure + start,
    )


def refinement_arrays(
    token_ids: np.ndarray,
    *,
    parent_stride: int,
    child_stride: int,
    mask_id: int,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(input_ids, target_ids, loss_mask)`` for one refinement step.

    The input reveals the parent grid (multiples of ``parent_stride``) as ground
    truth and masks every other slot. The loss covers the slots newly revealed at
    ``child_stride`` that are not padding.
    """

    length = int(token_ids.shape[0])
    inputs = np.full((length,), mask_id, dtype=np.int64)
    for position in revealed_positions(length, parent_stride):
        inputs[position] = token_ids[position]
    loss_mask = np.zeros((length,), dtype=bool)
    for position in refinement_positions(length, parent_stride, child_stride):
        if token_ids[position] != pad_id:
            loss_mask[position] = True
            inputs[position] = mask_id
    return inputs, token_ids.copy(), loss_mask


def split_windows(
    windows: Sequence[Window],
) -> dict[str, list[Window]]:
    grouped: dict[str, list[Window]] = {"train": [], "validation": [], "test": []}
    for window in windows:
        grouped.setdefault(window.split, []).append(window)
    return grouped


def movement_holdout(
    progressions: Sequence["HarmonyProgression"],
    *,
    seed: int,
) -> list["HarmonyProgression"]:
    """Reassign splits so every lineage appears in train (seen-composer test).

    Within each lineage the scores are shuffled deterministically; the first is
    held out for test, the second for validation, and the rest go to train. This
    isolates whether the operator can learn harmonic refinement for a composer it
    has seen (unseen movement), separate from cross-composer generalization.
    """

    import numpy as _np
    from collections import defaultdict
    from dataclasses import replace

    by_lineage: dict[str, list["HarmonyProgression"]] = defaultdict(list)
    for progression in progressions:
        by_lineage[progression.lineage_id].append(progression)
    rng = _np.random.default_rng(seed)
    out: list["HarmonyProgression"] = []
    for lineage in sorted(by_lineage):
        scores = sorted(by_lineage[lineage], key=lambda item: item.score_id)
        order = list(rng.permutation(len(scores)))
        for rank, index in enumerate(order):
            if rank == 0 and len(scores) >= 3:
                split = "test"
            elif rank == 1 and len(scores) >= 3:
                split = "validation"
            else:
                split = "train"
            out.append(replace(scores[index], split=split))
    return out
