"""Proposed model: sinusoidal encoding of the *real* measurement time."""
import math

import torch
import torch.nn as nn

from .common import LongitudinalTransformer, SingleFeatureTransformer


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal encoding evaluated at real (possibly irregularly spaced) times.

    Identical to the usual sinusoidal positional encoding except that the position
    is the measurement time itself rather than the index of the visit.
    """

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x, time):
        # x: [N, T, D], time: [N, T]
        N, T = time.shape
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = time.unsqueeze(-1) * div_term  # [N, T, D/2]
        pe = torch.zeros(N, T, self.d_model, device=x.device)
        pe[..., 0::2] = torch.sin(angles)
        pe[..., 1::2] = torch.cos(angles)
        return x + pe


class ProposedTransformer(LongitudinalTransformer):
    """Value embedding + continuous-time encoding, added at observed time points only."""

    def __init__(self, n_features, d_model, num_layers, max_len=None, nhead_time=2, dropout=0.2):
        super().__init__(n_features, d_model)
        self.feature_projection = nn.Linear(1, d_model)
        self.pos_encoder = ContinuousTimeEncoding(d_model)
        self._build_head(SingleFeatureTransformer(d_model, num_layers, nhead_time, dropout), dropout)

    def embed(self, values, time_flat, feature_mask_flat):
        B, F, T, _ = values.shape
        h = self.feature_projection(values).reshape(B * F, T, self.d_model)
        h_pe = self.pos_encoder(h, time_flat)
        return torch.where(feature_mask_flat.unsqueeze(-1), h_pe, h)
