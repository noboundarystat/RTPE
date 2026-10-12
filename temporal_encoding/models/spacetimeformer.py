"""Spacetimeformer-style embedding (Grigsby et al., 2021): value + Time2Vec -> projection,
plus learned position and variable embeddings. The same shared temporal encoder is used."""
import torch
import torch.nn as nn

from .common import LongitudinalTransformer, SingleFeatureTransformer


class Time2Vec(nn.Module):
    """One linear term plus (out_dim - 1) learned sinusoids of the real time."""

    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim

        self.linear = nn.Linear(1, 1, bias=True)
        self.omega = nn.Parameter(torch.randn(out_dim - 1) * 0.01)
        self.phi = nn.Parameter(torch.zeros(out_dim - 1))

        nn.init.uniform_(self.linear.weight, -0.01, 0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, time_seq):
        # time_seq: [N, T] -> [N, T, out_dim]
        t = time_seq.unsqueeze(-1)
        linear_part = self.linear(t)
        periodic_part = torch.sin(t * self.omega.view(1, 1, -1) + self.phi.view(1, 1, -1))
        return torch.cat([linear_part, periodic_part], dim=-1)


class Spacetimeformer(LongitudinalTransformer):
    def __init__(self, n_features, d_model, num_layers, max_len=5000,
                 time2vec_output_dim=8, nhead_time=2, dropout=0.2):
        super().__init__(n_features, d_model)
        self.max_len = max_len

        self.time2vec = Time2Vec(out_dim=time2vec_output_dim)
        self.vt_projection = nn.Linear(1 + time2vec_output_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.var_emb = nn.Embedding(n_features, d_model)
        self._build_head(SingleFeatureTransformer(d_model, num_layers, nhead_time, dropout), dropout)

    def embed(self, values, time_flat, feature_mask_flat):
        B, F, T, _ = values.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}.")
        device = values.device

        vt = torch.cat([values.reshape(B * F, T, 1), self.time2vec(time_flat)], dim=-1)
        h = self.vt_projection(vt)

        pos_idx = torch.arange(T, device=device).unsqueeze(0).expand(B * F, -1)
        var_idx = torch.arange(F, device=device).view(1, F, 1).expand(B, F, T).reshape(B * F, T)
        return h + self.pos_emb(pos_idx) + self.var_emb(var_idx)
