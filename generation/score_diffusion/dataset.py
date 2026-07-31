"""Window dataset over the full whole-score corpus for self-supervised denoising.

Reuses the *lineage* split that Ruby Partitura already froze in
``outputs/datasets/whole_score/<id>/manifest.json`` (each record carries a ``split`` of
``train``/``validation``/``test``), so train and held-out scores never share a lineage.

Every score is cut into meter-normalized windows (:func:`generation.score_diffusion.roll`).
Windows are stored *sparsely* -- only the active ``(channel, step)`` coordinates, which is
~0.2% of the grid -- and expanded to a dense ``(CHANNELS, STEPS)`` tensor on access, so the
whole 110-score corpus stays small in memory.

Diffusion works in ``[-1, 1]``: :func:`to_x0` maps a ``{0,1}`` roll to ``x0`` and
:func:`from_x0` inverts it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from generation.score_diffusion import roll as R

# Manifest ``split`` values, normalized to the three canonical names.
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "validation": "validation",
    "valid": "validation",
    "test": "test",
    "holdout": "test",
    "held_out": "test",
}


def to_x0(roll01: np.ndarray) -> np.ndarray:
    """{0,1} onset roll -> diffusion target in [-1, 1]."""
    return roll01.astype(np.float32) * 2.0 - 1.0


def from_x0(x0: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """[-1, 1] field -> {0,1} onset roll by thresholding at ``threshold``."""
    return (np.asarray(x0) > threshold).astype(np.float32)


def default_manifest() -> Path:
    return Path("outputs/datasets/whole_score/pilot_v1/manifest.json")


class WholeScoreRollDataset(Dataset):
    """Meter-normalized roll windows for one manifest split.

    Parameters
    ----------
    manifest_path : path to the whole-score dataset manifest.
    split : ``train`` | ``validation`` | ``test`` (aliases accepted).
    n_measures / stride : window geometry (stride defaults to ``n_measures``: no overlap).
    min_active : drop windows with fewer than this many onsets (skips empty rests/coda tails).
    """

    def __init__(self, manifest_path=None, split="train", *, n_measures=R.WINDOW_MEASURES,
                 stride=None, min_active=8):
        self.manifest_path = Path(manifest_path) if manifest_path else default_manifest()
        self.split = SPLIT_ALIASES.get(split, split)
        self.n_measures = n_measures
        self.stride = stride if stride is not None else n_measures
        self.steps = n_measures * R.SUBDIV
        manifest = json.loads(self.manifest_path.read_text())
        base = self.manifest_path.parent
        self.records = [r for r in manifest["records"]
                        if SPLIT_ALIASES.get(r.get("split"), r.get("split")) == self.split]
        # Sparse windows: (record_index, start_measure, channel_idx array, step_idx array).
        self._windows: list = []
        self.lineages: set = set()
        self._fallback_parts = 0
        self._total_parts = 0
        for ri, rec in enumerate(self.records):
            score = json.loads((base / rec["observation_file"]).read_text())["score"]
            fams = R.part_families(score)
            self._total_parts += len(fams)
            self._fallback_parts += self._count_true_fallbacks(score, fams)
            for start in R.score_windows(score, n_measures=n_measures, stride=self.stride):
                grid = R.build_window_roll(score, start, families=fams, n_measures=n_measures)
                ch, st = np.where(grid > 0)
                if ch.size < min_active:
                    continue
                self._windows.append((ri, start, ch.astype(np.int32), st.astype(np.int32)))
                self.lineages.add(rec.get("lineage_id"))

    @staticmethod
    def _count_true_fallbacks(score, fams) -> int:
        """Parts assigned to the fallback family with no keyword actually matching."""
        n = 0
        for part in score.get("parts", []):
            names = [part.get("name", ""), part.get("abbreviation", "")]
            for inst in part.get("instruments", []) or []:
                names.append(inst.get("name", ""))
            blob = "".join(x for x in names if x).lower()
            blob = "".join(c for c in blob if c.isalnum())
            matched = any(k in blob for k, _ in R._FAMILY_KEYWORDS)
            if not matched:
                n += 1
        return n

    def __len__(self) -> int:
        return len(self._windows)

    def roll01(self, index: int) -> np.ndarray:
        """Dense {0,1} roll for window ``index``."""
        _, _, ch, st = self._windows[index]
        grid = np.zeros((R.CHANNELS, self.steps), dtype=np.float32)
        grid[ch, st] = 1.0
        return grid

    def window_meta(self, index: int) -> dict:
        ri, start, ch, _ = self._windows[index]
        rec = self.records[ri]
        return {
            "record_index": ri,
            "start_measure": start,
            "lineage_id": rec.get("lineage_id"),
            "source_id": rec.get("source_id"),
            "n_active": int(ch.size),
        }

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(to_x0(self.roll01(index)))

    # --- corpus-level statistics ------------------------------------------------------
    def channel_marginals(self) -> np.ndarray:
        """Per-channel onset frequency across all windows -- ``(CHANNELS,)`` in [0, 1]."""
        acc = np.zeros(R.CHANNELS, dtype=np.float64)
        for i in range(len(self)):
            _, _, ch, _ = self._windows[i]
            np.add.at(acc, ch, 1.0)
        # Each window contributes ``steps`` cells per channel.
        return acc / max(1, len(self) * self.steps)

    def coverage(self) -> dict:
        return {
            "split": self.split,
            "records": len(self.records),
            "windows": len(self),
            "lineages": len({l for l in self.lineages if l is not None}),
            "parts": self._total_parts,
            "fallback_parts": self._fallback_parts,
            "mean_density": float(np.mean([self._windows[i][2].size for i in range(len(self))])
                                  / (R.CHANNELS * self.steps)) if len(self) else 0.0,
        }


def iter_batches(dataset, batch_size, *, shuffle=True, generator=None):
    """Yield stacked ``(B, CHANNELS, STEPS)`` x0 batches from a dataset."""
    n = len(dataset)
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size].tolist()
        yield torch.stack([dataset[j] for j in idx], 0)
