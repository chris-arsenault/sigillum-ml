"""Whole-score self-supervised diffusion (Phase 10).

A label-free, multi-track roll denoiser trained on the full Partitura corpus:
start from noise, iteratively refine toward a plausible score.
"""
from generation.score_diffusion import baselines, dataset, evaluate, model, roll

__all__ = ["roll", "dataset", "model", "baselines", "evaluate"]
