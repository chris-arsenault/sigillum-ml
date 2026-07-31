"""Non-learned reconstruction baselines -- the honesty gate for the denoiser.

Each baseline reconstructs the masked region of an onset roll given only the surrounding
context. The learned denoiser has to *beat* these to be worth anything:

* :class:`SilenceBaseline` -- predict nothing. Since the roll is ~99.8% empty, silence already
  scores very high accuracy; this is why we report active-cell F1, not accuracy.
* :class:`MarginalBaseline` -- predict the corpus-frequent cells (per-channel onset marginals
  thresholded). Context-independent; captures "which instruments/pitches are common".
* :class:`PersistenceBaseline` -- copy the equivalent slice of immediately preceding context
  (time-copy). Exploits the strong local repetition of orchestral music; the hardest baseline.

A baseline takes ``context`` (the {0,1} roll with the masked cells zeroed) and ``mask`` (``1`` on
the cells to reconstruct) and returns a full {0,1} roll; the harness scores only the masked cells.
"""
from __future__ import annotations

import numpy as np

from generation.score_diffusion.roll import CHANNELS


class SilenceBaseline:
    name = "silence"

    def predict(self, context: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.zeros_like(context)


class MarginalBaseline:
    """Predict cells whose per-channel corpus onset frequency clears a threshold.

    The threshold is calibrated (see :meth:`calibrate`) so the predicted density in the masked
    region roughly matches the corpus density -- otherwise the baseline could trivially maximize
    recall by predicting every channel.
    """

    name = "marginal"

    def __init__(self, marginals: np.ndarray, threshold: float):
        self.marginals = np.asarray(marginals, dtype=np.float64)
        self.threshold = float(threshold)

    @classmethod
    def calibrate(cls, marginals: np.ndarray, target_density: float) -> "MarginalBaseline":
        """Pick the highest threshold whose selected channels' total frequency ~ target density."""
        marg = np.asarray(marginals, dtype=np.float64)
        # Predicted density if we activate channels with marginal >= thr, for every masked step,
        # equals (number of selected channels) / CHANNELS. Choose thr to match target_density.
        order = np.sort(marg)[::-1]
        k = max(1, int(round(target_density * CHANNELS)))
        k = min(k, len(order))
        threshold = float(order[k - 1])
        return cls(marg, threshold)

    def predict(self, context: np.ndarray, mask: np.ndarray) -> np.ndarray:
        active_channels = (self.marginals >= self.threshold).astype(np.float32)
        pred = np.repeat(active_channels[:, None], context.shape[1], axis=1)
        return pred


class PersistenceBaseline:
    """Time-copy: reconstruct a masked step from the step ``period`` columns earlier.

    ``period`` defaults to the width of the masked region, so a masked block is filled with the
    block immediately before it. Cells with no valid source (out of range) stay silent.
    """

    name = "persistence"

    def __init__(self, period: int = None):
        self.period = period

    def predict(self, context: np.ndarray, mask: np.ndarray) -> np.ndarray:
        steps = context.shape[1]
        masked_cols = np.where(mask.any(axis=0))[0]
        if masked_cols.size == 0:
            return np.zeros_like(context)
        period = self.period if self.period is not None else (masked_cols.max() - masked_cols.min() + 1)
        period = max(1, int(period))
        pred = np.zeros_like(context)
        for s in masked_cols:
            src = s - period
            if src >= 0:
                pred[:, s] = context[:, src]
        return pred


def default_baselines(marginals: np.ndarray, target_density: float) -> list:
    return [
        SilenceBaseline(),
        MarginalBaseline.calibrate(marginals, target_density),
        PersistenceBaseline(),
    ]
