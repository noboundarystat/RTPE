"""Shortformer-style position-infused attention (Press et al., 2021).

Positions are injected into the queries and keys of every attention layer
(Q = Q_token + Q_pos, K = K_token + K_pos, V = V_token) instead of being added
to the input embeddings.
"""
import math

import torch
import torch.nn as nn
from torch.nn.functional import multi_head_attention_forward

from .common import LongitudinalTransformer


class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        return self.pe[:seq_len, :]


class PositionInfusedAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead=2, dim_feedforward=None, dropout=0.2):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")
        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)

        self.W_q_token = nn.Linear(d_model, d_model)
        self.W_k_token = nn.Linear(d_model, d_model)
        self.W_v_token = nn.Linear(d_model, d_model)
        self.W_q_pos = nn.Linear(d_model, d_model)
        self.W_k_pos = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, x, pos_embed, src_key_padding_mask=None):
        """x [N, T, D], pos_embed [T, D], src_key_padding_mask [N, T] (True = ignore)."""
        bsz, seq_len, _ = x.shape
        pos_embed = pos_embed.to(device=x.device, dtype=x.dtype)
        pos_embed = pos_embed.unsqueeze(0).expand(bsz, seq_len, -1)

        Q = self.W_q_token(x) + self.W_q_pos(pos_embed)
        K = self.W_k_token(x) + self.W_k_pos(pos_embed)
        V = self.W_v_token(x)

        # The token/position projections above already produced Q, K, V, so the
        # functional MHA is given identity input projections.
        eye = torch.eye(self.d_model, device=x.device, dtype=x.dtype)
        attn_output, _ = multi_head_attention_forward(
            query=Q.transpose(0, 1),
            key=K.transpose(0, 1),
            value=V.transpose(0, 1),
            embed_dim_to_check=self.d_model,
            num_heads=self.nhead,
            in_proj_weight=None,
            in_proj_bias=None,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.dropout.p,
            out_proj_weight=self.out_proj.weight,
            out_proj_bias=self.out_proj.bias,
            training=self.training,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
            attn_mask=None,
            use_separate_proj_weight=True,
            q_proj_weight=eye,
            k_proj_weight=eye,
            v_proj_weight=eye,
        )

        x = self.norm1(x + self.dropout(attn_output.transpose(0, 1)))
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.norm2(x + self.dropout(x2))


class PositionInfusedEncoder(nn.Module):
    def __init__(self, d_model, num_layers, nhead=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            PositionInfusedAttentionLayer(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, pos_embed, mask):
        src_key_padding_mask = (mask == 0)
        for layer in self.layers:
            x = layer(x, pos_embed, src_key_padding_mask=src_key_padding_mask)
        return x


class Shortformer(LongitudinalTransformer):
    def __init__(self, n_features, d_model, num_layers, max_len=5000, nhead_time=2, dropout=0.2):
        super().__init__(n_features, d_model)
        self.max_len = max_len
        self.feature_projection = nn.Linear(1, d_model)
        self.pos_embedding = SinusoidalPosEmbedding(d_model, max_len)
        self._build_head(PositionInfusedEncoder(d_model, num_layers, nhead_time, dropout), dropout)

    def embed(self, values, time_flat, feature_mask_flat):
        B, F, T, _ = values.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length T={T} exceeds max_len={self.max_len}.")
        return self.feature_projection(values).reshape(B * F, T, self.d_model)

    def encode(self, h, attn_mask_flat):
        pos_embed = self.pos_embedding(h.shape[1])  # [T, D], shared by all sequences
        return self.feature_transformer(h, pos_embed, attn_mask_flat)
