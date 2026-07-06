"""Factored ThemeGPT (v1): per-field embeddings summed in, per-field prediction heads out.

Same transformer backbone as v0, but the 4,170-word token embedding/head is replaced by a handful
of tiny field embeddings/heads — cutting ~2M parameters and letting the model share musical
primitives across notes. Conditioning fields (`cond`) are given, not predicted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from generation.theme_nn.vocab_factored import FIELDS, KIND

PRED = ["kind", "degree", "alt", "oct", "dur"]   # fields the model predicts (cond is conditioning)


@dataclass
class FactoredConfig:
    field_sizes: dict
    block_size: int = 768
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1


class FactoredThemeGPT(nn.Module):
    def __init__(self, config: FactoredConfig):
        super().__init__()
        self.config = config
        self.emb = nn.ModuleDict({f: nn.Embedding(config.field_sizes[f], config.d_model) for f in FIELDS})
        self.pos = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.n_head, 4 * config.d_model, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.n_layer)
        self.ln = nn.LayerNorm(config.d_model)
        self.heads = nn.ModuleDict({f: nn.Linear(config.d_model, config.field_sizes[f]) for f in PRED})
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, positions: torch.Tensor) -> dict:
        # positions: (B, T, n_fields) long
        _, T, _ = positions.shape
        x = sum(self.emb[f](positions[:, :, i]) for i, f in enumerate(FIELDS))
        x = self.drop(x + self.pos(torch.arange(T, device=positions.device)))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=positions.device)
        x = self.ln(self.blocks(x, mask=mask, is_causal=True))
        return {f: self.heads[f](x) for f in PRED}

    @torch.no_grad()
    def generate(self, prefix: list[tuple], max_events: int, temperature: float = 1.0,
                 top_k: int = 0) -> list[tuple]:
        def sample(logits):
            logits = logits / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float("inf")
            return int(torch.multinomial(F.softmax(logits, dim=-1), 1))

        seq = list(prefix)
        for _ in range(max_events):
            idx = torch.tensor([seq], dtype=torch.long)[:, -self.config.block_size:]
            out = self(idx)
            kind = sample(out["kind"][0, -1])
            if kind == KIND["EOS"]:
                break
            if kind != KIND["EVENT"]:                 # model should only emit EVENT/EOS here; coerce
                kind = KIND["EVENT"]
            degree = sample(out["degree"][0, -1]); alt = sample(out["alt"][0, -1])
            octave = sample(out["oct"][0, -1]); dur = sample(out["dur"][0, -1])
            seq.append((kind, degree, alt, octave, dur, 0))
        return seq[len(prefix):]
