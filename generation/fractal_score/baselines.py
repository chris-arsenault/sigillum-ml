"""Non-learned refinement baselines the learned operator must beat.

These make the fractalization claim honest: if a learned operator cannot invert
a coarse harmonic grid into a finer one better than copying the nearest pillar or
replaying corpus statistics, the direction is not yet supported.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from generation.fractal_score.dataset import Window
from generation.fractal_score.ladder import refinement_positions, revealed_positions


class Baseline:
    """Fill masked child slots from parent context; no learning across a window."""

    name = "baseline"

    def fill(
        self,
        token_ids: np.ndarray,
        *,
        parent_stride: int,
        child_stride: int,
    ) -> np.ndarray:
        raise NotImplementedError


class CopyNearestPillar(Baseline):
    """Copy each masked slot from the nearest revealed parent slot to its left."""

    name = "copy_nearest_pillar"

    def fill(self, token_ids, *, parent_stride, child_stride):
        length = int(token_ids.shape[0])
        result = token_ids.copy()
        known = revealed_positions(length, parent_stride)
        for position in refinement_positions(length, parent_stride, child_stride):
            left = position - (position % parent_stride)
            result[position] = token_ids[left] if left in known else token_ids[known[0]]
        return result


@dataclass
class Unigram(Baseline):
    """Predict the single most frequent training chord for every masked slot."""

    most_common_id: int
    name: str = "unigram"

    @classmethod
    def fit(cls, windows: Iterable[Window], ignore_ids: Sequence[int]) -> "Unigram":
        counts: Counter[int] = Counter()
        ignore = set(ignore_ids)
        for window in windows:
            for value in window.token_ids.tolist():
                if value not in ignore:
                    counts[value] += 1
        if not counts:
            raise ValueError("unigram baseline saw no usable tokens")
        return cls(most_common_id=counts.most_common(1)[0][0])

    def fill(self, token_ids, *, parent_stride, child_stride):
        length = int(token_ids.shape[0])
        result = token_ids.copy()
        for position in refinement_positions(length, parent_stride, child_stride):
            result[position] = self.most_common_id
        return result


@dataclass
class Bigram(Baseline):
    """Predict each masked slot from the nearest revealed left neighbour's transition."""

    transitions: dict[int, int]
    fallback_id: int
    name: str = "bigram"

    @classmethod
    def fit(cls, windows: Iterable[Window], ignore_ids: Sequence[int]) -> "Bigram":
        pair_counts: dict[int, Counter[int]] = defaultdict(Counter)
        unigram: Counter[int] = Counter()
        ignore = set(ignore_ids)
        for window in windows:
            values = window.token_ids.tolist()
            for previous, current in zip(values, values[1:]):
                if previous in ignore or current in ignore:
                    continue
                pair_counts[previous][current] += 1
                unigram[current] += 1
        if not unigram:
            raise ValueError("bigram baseline saw no usable transitions")
        transitions = {
            previous: counter.most_common(1)[0][0]
            for previous, counter in pair_counts.items()
        }
        return cls(transitions=transitions, fallback_id=unigram.most_common(1)[0][0])

    def fill(self, token_ids, *, parent_stride, child_stride):
        length = int(token_ids.shape[0])
        result = token_ids.copy()
        for position in refinement_positions(length, parent_stride, child_stride):
            left = position - (position % parent_stride)
            previous = int(result[left]) if left >= 0 else self.fallback_id
            result[position] = self.transitions.get(previous, self.fallback_id)
        return result


@dataclass
class BigramLM:
    """A smoothed bigram language model, for a fair held-out likelihood baseline.

    Unlike the argmax ``Bigram`` filler, this scores the probability of the true
    next chord given the true previous chord. It is the natural probabilistic
    baseline the learned operator's held-out negative log-likelihood must beat.
    """

    log_probs: np.ndarray
    unigram_log: np.ndarray

    @classmethod
    def fit(
        cls,
        windows: Iterable[Window],
        vocab_size: int,
        ignore_ids: Sequence[int],
        alpha: float = 0.5,
    ) -> "BigramLM":
        pair = np.full((vocab_size, vocab_size), alpha, dtype=np.float64)
        uni = np.full((vocab_size,), alpha, dtype=np.float64)
        ignore = set(ignore_ids)
        for window in windows:
            values = window.token_ids.tolist()
            for value in values:
                if value not in ignore:
                    uni[value] += 1.0
            for previous, current in zip(values, values[1:]):
                if previous in ignore or current in ignore:
                    continue
                pair[previous, current] += 1.0
        log_probs = np.log(pair / pair.sum(axis=1, keepdims=True))
        unigram_log = np.log(uni / uni.sum())
        return cls(log_probs=log_probs, unigram_log=unigram_log)

    def log_prob(self, previous: int, current: int) -> float:
        return float(self.log_probs[previous, current])
