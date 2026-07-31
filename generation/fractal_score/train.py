"""Training loop for the shared harmonic refinement operator."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from generation.fractal_score.dataset import Window, refinement_arrays
from generation.fractal_score.ladder import RefinementSchedule
from generation.fractal_score.model import RefinementConfig, RefinementOperator
from generation.fractal_score.vocab import HarmonyVocab


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 20260731

    def to_dict(self) -> dict[str, float | int]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
        }


def _collate(
    windows: Sequence[Window],
    schedule: RefinementSchedule,
    vocab: HarmonyVocab,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = schedule.steps()
    inputs, targets, masks, levels, keys = [], [], [], [], []
    for window in windows:
        parent_stride, child_stride = steps[int(rng.integers(len(steps)))]
        row_in, row_target, row_mask = refinement_arrays(
            window.token_ids,
            parent_stride=parent_stride,
            child_stride=child_stride,
            mask_id=vocab.mask_id,
            pad_id=vocab.pad_id,
        )
        inputs.append(row_in)
        targets.append(row_target)
        masks.append(row_mask)
        levels.append(schedule.level_index(child_stride))
        keys.append(window.key_id)
    return (
        np.stack(inputs),
        np.stack(targets),
        np.stack(masks),
        np.array(levels, dtype=np.int64),
        np.array(keys, dtype=np.int64),
    )


def train_operator(
    train_windows: Sequence[Window],
    schedule: RefinementSchedule,
    vocab: HarmonyVocab,
    model_config: RefinementConfig,
    train_config: TrainConfig,
    device: torch.device | None = None,
) -> tuple[RefinementOperator, list[dict[str, float]]]:
    device = device or torch.device("cpu")
    torch.manual_seed(train_config.seed)
    rng = np.random.default_rng(train_config.seed)
    model = RefinementOperator(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    loss_fn = nn.CrossEntropyLoss()
    windows = list(train_windows)
    history: list[dict[str, float]] = []
    for epoch in range(train_config.epochs):
        rng.shuffle(windows)  # type: ignore[arg-type]
        model.train()
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(windows), train_config.batch_size):
            batch = windows[start : start + train_config.batch_size]
            inputs, targets, masks, levels, keys = _collate(batch, schedule, vocab, rng)
            if not masks.any():
                continue
            input_tensor = torch.from_numpy(inputs).to(device)
            target_tensor = torch.from_numpy(targets).to(device)
            mask_tensor = torch.from_numpy(masks).to(device)
            level_tensor = torch.from_numpy(levels).to(device)
            key_tensor = torch.from_numpy(keys).to(device)
            pad_mask = input_tensor == vocab.pad_id
            logits = model(input_tensor, level_tensor, key_tensor, pad_mask=pad_mask)
            selected = mask_tensor.reshape(-1)
            flat_logits = logits.reshape(-1, logits.size(-1))[selected]
            flat_targets = target_tensor.reshape(-1)[selected]
            loss = loss_fn(flat_logits, flat_targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(
            {"epoch": epoch, "loss": epoch_loss / batches if batches else 0.0}
        )
    return model, history
