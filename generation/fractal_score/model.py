"""The shared recursive refinement operator (a small conditional Transformer).

One operator is reused at every ladder level. It reads a partly revealed grid
(true chords at the parent stride, ``<mask>`` elsewhere), the level being
refined, and the home key, and predicts a chord for every slot. Because the same
weights refine coarse-to-fine at each stride, applying it repeatedly is the
fractalization step: a coarse skeleton is engraved into progressively finer
detail by the same learned primitive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class RefinementConfig:
    vocab_size: int
    key_count: int
    level_count: int
    max_length: int = 128
    d_model: int = 192
    layers: int = 4
    heads: int = 6
    ff: int = 512
    dropout: float = 0.1

    def to_dict(self) -> dict[str, int | float]:
        return {
            "vocab_size": self.vocab_size,
            "key_count": self.key_count,
            "level_count": self.level_count,
            "max_length": self.max_length,
            "d_model": self.d_model,
            "layers": self.layers,
            "heads": self.heads,
            "ff": self.ff,
            "dropout": self.dropout,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "RefinementConfig":
        return cls(**{key: value[key] for key in value if key in cls.__annotations__})


class RefinementOperator(nn.Module):
    def __init__(self, config: RefinementConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        self.level_embedding = nn.Embedding(config.level_count, config.d_model)
        self.key_embedding = nn.Embedding(max(config.key_count, 1), config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.heads,
            dim_feedforward=config.ff,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        level_id: torch.Tensor,
        key_id: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.level_embedding(level_id).unsqueeze(1)
            + self.key_embedding(key_id).unsqueeze(1)
        )
        hidden = hidden * math.sqrt(1.0)
        if pad_mask is not None and not bool(pad_mask.any()):
            pad_mask = None
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        hidden = self.norm(hidden)
        return self.head(hidden)
