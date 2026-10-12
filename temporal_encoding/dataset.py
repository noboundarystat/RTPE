"""PyTorch dataset for the wide-format simulated longitudinal data."""
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


def _feature_sort_key(col_name):
    """Sort x1, x2, ..., x10 numerically rather than lexicographically."""
    if col_name.startswith("x") and col_name[1:].isdigit():
        return int(col_name[1:])
    return col_name


class LongitudinalDataset(Dataset):
    """
    Args:
        medical_df: columns [subject, x1..xM, Y]; each xj holds a length-T array.
        time_df:    column ``time`` holding each subject's measurement times.
        max_len:    sequences are padded (with 0) / truncated to this length.
        data_mask:  optional bool array [N, max_len, M], True = observed.
        scalers:    per-feature StandardScalers fitted on the training set.
        fit_scalers: fit ``scalers`` on the *observed* values of this dataset (training set only).

    Items are ``(features [M, T], time [T], pad_mask [T], value_mask [M, T], label)``
    when ``data_mask`` is given, otherwise without ``value_mask``.
    """

    def __init__(self, medical_df, time_df, max_len=21, data_mask=None,
                 scalers=None, fit_scalers=False):
        self.medical_df = medical_df.reset_index(drop=True)
        self.time_df = time_df.reset_index(drop=True)
        self.max_len = max_len

        self.feature_cols = sorted(
            [c for c in self.medical_df.columns if c.startswith("x")], key=_feature_sort_key
        )
        self.M = len(self.feature_cols)

        self.has_missing = data_mask is not None
        if self.has_missing:
            assert data_mask.shape == (len(self.medical_df), self.max_len, self.M)
            self.data_mask = data_mask.astype(bool)
        else:
            self.data_mask = None

        if scalers is None:
            scalers = {col: StandardScaler() for col in self.feature_cols}
        self.scalers = scalers

        if fit_scalers:
            self._fit_scalers()

    def _fit_scalers(self):
        for j, col in enumerate(self.feature_cols):
            vals = []
            for i in range(len(self.medical_df)):
                seq = np.asarray(self.medical_df.iloc[i][col], dtype=float)
                L = min(len(seq), self.max_len)
                if self.has_missing:
                    obs = self.data_mask[i, :L, j]
                    if np.any(obs):
                        vals.append(seq[:L][obs])
                else:
                    vals.append(seq[:L])

            flat = np.concatenate(vals, axis=0) if vals else np.array([0.0])
            if flat.size == 0:
                flat = np.array([0.0])
            self.scalers[col].fit(flat.reshape(-1, 1))

    def __len__(self):
        return len(self.medical_df)

    def __getitem__(self, idx):
        first_seq = np.asarray(self.medical_df.iloc[idx][self.feature_cols[0]], dtype=float)
        L = min(len(first_seq), self.max_len)

        pad_mask = np.ones(self.max_len, dtype=float)
        pad_mask[L:] = 0.0

        time_seq = np.asarray(self.time_df.iloc[idx]["time"], dtype=float)[:L]
        time_seq = np.pad(time_seq, (0, self.max_len - L), "constant", constant_values=0.0)

        feature_mat = np.zeros((self.M, self.max_len), dtype=float)
        value_mask = np.zeros((self.M, self.max_len), dtype=float)

        for j, col in enumerate(self.feature_cols):
            seq = np.asarray(self.medical_df.iloc[idx][col], dtype=float)[:L]
            seq_scaled = self.scalers[col].transform(seq.reshape(-1, 1)).flatten()

            if self.has_missing:
                obs = self.data_mask[idx, :L, j]
                seq_scaled = seq_scaled.copy()
                seq_scaled[~obs] = 0.0
                value_mask[j, :L] = obs.astype(float)

            feature_mat[j] = np.pad(seq_scaled, (0, self.max_len - L), "constant", constant_values=0.0)

        label = float(self.medical_df.iloc[idx]["Y"])

        item = [
            torch.FloatTensor(feature_mat),
            torch.FloatTensor(time_seq),
            torch.FloatTensor(pad_mask),
        ]
        if self.has_missing:
            item.append(torch.FloatTensor(value_mask))
        item.append(torch.FloatTensor([label]))
        return tuple(item)
