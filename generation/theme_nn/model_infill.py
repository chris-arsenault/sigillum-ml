"""Encoder-decoder for kernel infilling: the encoder reads the kernel, the decoder fills the spans.

Factored field embeddings (shared encoder/decoder), a Transformer encoder over the kernel, and a
causal Transformer decoder that cross-attends to it and predicts each field of the gap fills. This
is the designed model — fill-between-pins — not a free-running decoder-only LM.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from generation.theme_nn.infill import FIELDS

# Diatonic-interval-primary: the decoder predicts KIND, the DIATONIC-STEP interval (the move along
# the scale — pitch is the running scale position, so it lands on a scale tone by construction), the
# ALTERATION (chromatic accidental, per-note so it can't drift), and DURATION. Absolute degree/oct
# are input-only context (the model conditions on where it is, predicts how to move) — what the
# coherent Markov does, without the absolute-pitch lock and without the chromatic drift.
PRED = ["kind", "dstep", "alt", "dur"]
AUX = ["fig", "motif", "cpos"]                   # étude targets: predicted to shape the trunk, never fed


@dataclass
class InfillConfig:
    field_sizes: dict
    enc_block: int = 384
    dec_block: int = 384
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.1
    n_fig: int = 0       # figuration classes (étude)
    n_motif: int = 0     # motif-transform classes (étude)
    n_cpos: int = 0      # chord-position classes (étude)
    n_dstep: int = 0     # diatonic-step interval classes (the primary pitch head)


class FactoredEncDec(nn.Module):
    def __init__(self, cfg: InfillConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.ModuleDict({f: nn.Embedding(cfg.field_sizes[f], cfg.d_model) for f in FIELDS})
        self.enc_pos = nn.Embedding(cfg.enc_block, cfg.d_model)
        self.dec_pos = nn.Embedding(cfg.dec_block, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        enc_layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_head, 4 * cfg.d_model, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, cfg.n_layer)
        dec_layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.n_head, 4 * cfg.d_model, cfg.dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, cfg.n_layer)
        self.ln = nn.LayerNorm(cfg.d_model)
        pred_sizes = {**cfg.field_sizes, "dstep": cfg.n_dstep}   # dstep is a head, not an input field
        self.heads = nn.ModuleDict({f: nn.Linear(cfg.d_model, pred_sizes[f]) for f in PRED})
        aux_sizes = {"fig": cfg.n_fig, "motif": cfg.n_motif, "cpos": cfg.n_cpos}
        self.aux = nn.ModuleDict({f: nn.Linear(cfg.d_model, aux_sizes[f])
                                  for f in AUX if aux_sizes[f] > 0})
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed(self, positions, pos_emb):
        T = positions.shape[1]
        x = sum(self.emb[f](positions[:, :, i]) for i, f in enumerate(FIELDS))
        return self.drop(x + pos_emb(torch.arange(T, device=positions.device)))

    def encode(self, enc, enc_pad=None):
        return self.encoder(self._embed(enc, self.enc_pos), src_key_padding_mask=enc_pad)

    def decode(self, dec, memory, enc_pad=None, dec_pad=None):
        T = dec.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=dec.device)
        out = self.decoder(self._embed(dec, self.dec_pos), memory, tgt_mask=mask, tgt_is_causal=True,
                           memory_key_padding_mask=enc_pad, tgt_key_padding_mask=dec_pad)
        h = self.ln(out)
        return {**{f: self.heads[f](h) for f in PRED}, **{f: self.aux[f](h) for f in self.aux}}

    def forward(self, enc, dec, enc_pad=None, dec_pad=None):
        return self.decode(dec, self.encode(enc, enc_pad), enc_pad, dec_pad)
