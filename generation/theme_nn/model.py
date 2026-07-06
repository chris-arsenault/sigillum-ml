"""A small decoder-only Transformer (the baseline note-predictor) over the token sequence.

Standard nanoGPT-shaped causal LM built on torch.nn — no custom CUDA, trains on CPU. This is the
flat baseline the deferred motif-developer is meant to beat; it intentionally has no infilling or
constrained decode yet (those come once the baseline trains and we can listen).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1


class ThemeGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok = nn.Embedding(config.vocab_size, config.d_model)
        self.pos = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_head, 4 * config.d_model, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.n_layer)
        self.ln = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        return self.head(self.ln(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, eos: int, temperature: float = 1.0,
                 top_k: int | None = 40) -> torch.Tensor:
        for _ in range(max_new):
            logits = self(idx[:, -self.config.block_size:])[:, -1, :] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == eos).all():
                break
        return idx
