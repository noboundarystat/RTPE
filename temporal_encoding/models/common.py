"""Building blocks shared by every neural model in the comparison.

All models use the same skeleton so that differences in test error can be
attributed to how time/position is encoded:

    value -> embedding (+ time/position information)
          -> temporal encoder, shared across features, run per (subject, feature)
          -> masked mean pooling over time
          -> concatenate features -> fusion MLP -> cross-feature MLP -> regression head

The per-feature loop is vectorised by folding the feature axis into the batch
axis (``[B, F, T] -> [B*F, T]``). Attention never mixes across the batch axis,
so this is mathematically identical to looping over features.
"""
import torch
import torch.nn as nn


def masked_mean_pool(h, mask):
    """Mean over time of ``h`` [N, T, D], counting only positions where ``mask`` [N, T] is True."""
    w = mask.unsqueeze(-1).type_as(h)
    denom = w.sum(dim=1).clamp_min(1.0)
    return (h * w).sum(dim=1) / denom


def build_feature_masks(mask, value_mask, n_features):
    """Build the masks used by the temporal encoder.

    Args:
        mask:       [B, T], 1 = real time point, 0 = padding.
        value_mask: [B, F, T] or None, 1 = value observed, 0 = missing.

    Returns:
        feature_mask:      [B, F, T] bool, valid time point AND observed value.
        feature_mask_flat: [B*F, T] bool, same thing with features folded into the batch.
        attn_mask_flat:    [B*F, T] bool, like ``feature_mask_flat`` except that a
                           (subject, feature) trajectory with *no* observed point falls back
                           to its valid time points. An all-masked ``src_key_padding_mask``
                           makes attention softmax return NaN; the pooled output of such a
                           trajectory is zeroed afterwards via ``no_observed_flat``.
        no_observed_flat:  [B*F] bool, trajectories with no observed point.
    """
    B, T = mask.shape
    base_mask = mask.bool()

    if value_mask is not None:
        feature_mask = base_mask.unsqueeze(1) & value_mask.bool()
    else:
        feature_mask = base_mask.unsqueeze(1).expand(B, n_features, T)

    feature_mask_flat = feature_mask.reshape(B * n_features, T)
    attn_mask_flat = feature_mask_flat.clone()
    no_observed_flat = attn_mask_flat.sum(dim=1) == 0

    if no_observed_flat.any():
        base_mask_flat = base_mask.unsqueeze(1).expand(B, n_features, T).reshape(B * n_features, T)
        fallback = base_mask_flat[no_observed_flat].clone()

        no_valid_time = fallback.sum(dim=1) == 0
        if no_valid_time.any():
            fallback[no_valid_time, 0] = True

        attn_mask_flat[no_observed_flat] = fallback

    return feature_mask, feature_mask_flat, attn_mask_flat, no_observed_flat


class SingleFeatureTransformer(nn.Module):
    """Standard transformer encoder shared across all features (key-padding mask only)."""

    def __init__(self, d_model, num_layers, nhead=2, dropout=0.2):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask):
        # mask: [N, T], True/1 = attend, False/0 = ignore
        return self.transformer(x, src_key_padding_mask=(mask == 0))


class CrossFeatureInteraction(nn.Module):
    """Residual MLP applied to the fused subject representation."""

    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, fused_feature):
        return self.layer_norm(fused_feature + self.ffn(fused_feature))


class LongitudinalTransformer(nn.Module):
    """Skeleton shared by all models; subclasses supply the time/position handling.

    Inputs to ``forward``:
        x:          [B, F, T] standardized feature values
        time:       [B, T] real (continuous) measurement times
        mask:       [B, T] 1 = real time point, 0 = padding
        value_mask: [B, F, T] or None, 1 = observed, 0 = missing
    Output: [B] predicted (standardized) outcome.

    Subclasses must, in ``__init__``: call ``super().__init__(n_features, d_model)``, create
    their embedding modules, create a temporal encoder, then call ``_build_head``. They
    implement ``embed`` and may override ``encode``.
    """

    def __init__(self, n_features, d_model):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

    def _build_head(self, encoder, dropout):
        self.feature_transformer = encoder
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.n_features * self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cross_feature_interaction = CrossFeatureInteraction(self.d_model, dropout=dropout)
        # No activation on the output: targets are standardized and can be negative.
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 1),
        )

    def embed(self, values, time_flat, feature_mask_flat):
        """values [B, F, T, 1] (missing/padded entries already zeroed), time_flat [B*F, T],
        feature_mask_flat [B*F, T] -> embeddings [B*F, T, D]."""
        raise NotImplementedError

    def encode(self, h, attn_mask_flat):
        """[B*F, T, D] -> [B*F, T, D]."""
        return self.feature_transformer(h, attn_mask_flat)

    def forward(self, x, time, mask, value_mask=None):
        B, F, T = x.shape
        if F != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {F}")

        feature_mask, feature_mask_flat, attn_mask_flat, no_observed_flat = build_feature_masks(
            mask, value_mask, F
        )

        values = x.unsqueeze(-1).clone()
        values[~feature_mask.unsqueeze(-1)] = 0.0

        # Every feature of a subject shares that subject's measurement times.
        time_flat = time.unsqueeze(1).expand(B, F, T).reshape(B * F, T)

        h = self.embed(values, time_flat, feature_mask_flat)
        h = self.encode(h, attn_mask_flat)

        pooled = masked_mean_pool(h, feature_mask_flat)
        if no_observed_flat.any():
            pooled = pooled.masked_fill(no_observed_flat.unsqueeze(1), 0.0)

        fused = self.feature_fusion(pooled.reshape(B, F * self.d_model))
        enhanced = self.cross_feature_interaction(fused)
        return self.regressor(enhanced).squeeze(-1)
