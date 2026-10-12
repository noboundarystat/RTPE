"""Stochastic positional encoding (SPE; Liutkus et al., 2021), sine variant with gating.

Note: SPE draws fresh random realizations on every forward pass (also in eval
mode); this is part of the method, so predictions are stochastic.
"""
import math

import torch
import torch.nn as nn

from .common import LongitudinalTransformer, SingleFeatureTransformer


class SineSPE(nn.Module):
    def __init__(self, num_heads=1, in_features=64, num_realizations=64, num_sines=1):
        super().__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.num_realizations = num_realizations
        self.num_sines = num_sines

        for param in ["freqs", "offsets", "gains"]:
            self.register_parameter(
                param, nn.Parameter(torch.randn(num_heads, in_features, num_sines))
            )

        self.gains.data[...] /= torch.sqrt(self.gains.norm(dim=-1, keepdim=True)) / 2.0
        self.freqs.data[...] -= 4.0

        self.code_shape = (num_heads, in_features)

    def forward(self, shape, num_realizations=None):
        if len(shape) != 2:
            raise ValueError("SineSPE expects input shape (batch_size, seq_len).")

        seq_len = shape[1]
        indices = torch.linspace(0, seq_len - 1, seq_len, device=self.freqs.device)
        freqs = torch.sigmoid(self.freqs[:, :, None, :]) / 2.0

        phases_q = 2 * math.pi * freqs * indices[None, None, :, None] + self.offsets[:, :, None, :]
        omega_q = torch.stack([torch.cos(phases_q), torch.sin(phases_q)], dim=-1).view(
            1, self.num_heads, self.in_features, seq_len, 2 * self.num_sines
        )

        phases_k = 2 * math.pi * freqs * indices[None, None, :, None]
        omega_k = torch.stack([torch.cos(phases_k), torch.sin(phases_k)], dim=-1).view(
            1, self.num_heads, self.in_features, seq_len, 2 * self.num_sines
        )

        gains = nn.functional.softplus(self.gains)
        gains = torch.stack([gains, gains], dim=-1).view(
            self.num_heads, self.in_features, 2 * self.num_sines
        )

        if num_realizations is None:
            num_realizations = self.num_realizations

        z = torch.randn(
            1, self.num_heads, self.in_features, 2 * self.num_sines, num_realizations,
            device=self.freqs.device,
        ) / math.sqrt(self.num_sines * 2)
        z = z * gains[None, ..., None]

        qbar = torch.matmul(omega_q, z)
        kbar = torch.matmul(omega_k, z)

        qbar = qbar.permute(0, 3, 1, 2, 4)
        kbar = kbar.permute(0, 3, 1, 2, 4)

        scale = (num_realizations * self.in_features) ** 0.25
        return qbar / scale, kbar / scale


class SPEFilter(nn.Module):
    def __init__(self, gated=True, code_shape=None):
        super().__init__()
        self.gated = gated
        self.code_shape = code_shape

        if gated:
            if code_shape is None:
                raise RuntimeError("code_shape must be provided if gated=True.")
            self.register_parameter("gate", nn.Parameter(torch.randn(code_shape)))

    def forward(self, queries, keys, code):
        qbar, kbar = code

        if self.code_shape is not None and qbar.shape[-3:-1] != self.code_shape:
            raise ValueError(f"Shape mismatch: {qbar.shape[-3:-1]} vs {self.code_shape}.")

        if self.gated:
            in_features = qbar.shape[-2]
            num_realizations = qbar.shape[-1]
            gating_noise = torch.randn(
                self.code_shape + (num_realizations,), device=queries.device
            ) / (in_features * num_realizations) ** 0.25

            gate = torch.sigmoid(self.gate[..., None])
            qbar = torch.sqrt(1.0 - gate) * qbar + torch.sqrt(gate) * gating_noise
            kbar = torch.sqrt(1.0 - gate) * kbar + torch.sqrt(gate) * gating_noise

        qhat = (qbar * queries[..., None]).sum(dim=-2)
        khat = (kbar * keys[..., None]).sum(dim=-2)
        return qhat, khat


class SPETransformer(LongitudinalTransformer):
    def __init__(self, n_features, d_model, num_layers, max_len=None, nhead_time=2, dropout=0.2):
        super().__init__(n_features, d_model)
        self.feature_projection = nn.Linear(1, d_model)
        self.spe_encoder = SineSPE(
            num_heads=1, in_features=d_model, num_realizations=d_model, num_sines=5
        )
        self.spe_filter = SPEFilter(gated=True, code_shape=self.spe_encoder.code_shape)
        self._build_head(SingleFeatureTransformer(d_model, num_layers, nhead_time, dropout), dropout)

    def embed(self, values, time_flat, feature_mask_flat):
        B, F, T, _ = values.shape
        h = self.feature_projection(values).reshape(B * F, T, self.d_model)

        # SPE codes depend only on the sequence length -> shared by all B*F sequences.
        pos_codes = self.spe_encoder((1, T))
        queries = h.unsqueeze(2)  # [BF, T, 1, D]
        qhat, _ = self.spe_filter(queries, queries, pos_codes)
        return qhat.squeeze(2)
