"""Regular transformer: sinusoidal encoding of the visit *index*."""
import math

import torch
import torch.nn as nn

from .common import LongitudinalTransformer, SingleFeatureTransformer


class IndexPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (position = slot index 0..T-1)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class ClassicTransformer(LongitudinalTransformer):
    def __init__(self, n_features, d_model, num_layers, max_len=5000, nhead_time=2, dropout=0.2):
        super().__init__(n_features, d_model)
        self.feature_projection = nn.Linear(1, d_model)
        self.pos_encoder = IndexPositionalEncoding(d_model, max_len)
        self._build_head(SingleFeatureTransformer(d_model, num_layers, nhead_time, dropout), dropout)

    def embed(self, values, time_flat, feature_mask_flat):
        B, F, T, _ = values.shape
        h = self.feature_projection(values).reshape(B * F, T, self.d_model)
        return self.pos_encoder(h)
