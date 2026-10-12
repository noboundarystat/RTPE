"""ElasticNet baseline on the flattened (time x feature) design matrix.

Missing X values are filled by (single or multiple) imputation with sklearn's
IterativeImputer (BayesianRidge, posterior sampling); if the imputer fails, mean
imputation is used.
"""
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import numpy as np

# sklearn IterativeImputer import gate
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge


def train_and_evaluate_elasticnet(
        df_wide,
        train_idx,
        test_idx,
        l1_ratio=0.2,
        alpha=0.5,
        data_mask_full=None,
        random_state=42,
        n_imputations=1,
        imputer_max_iter=5,
        y_scaler=None,
):
    """
    ElasticNet with (optional) Multiple Imputation (MI) for missing X.

    Expected inputs:
      - df_wide: wide-format dataframe. Columns x1..xM each hold a length-T array, plus "Y".
                df_wide["Y"] is expected to be standardized already.
      - data_mask_full: shape (N, T, M), bool-like, True=observed, False=missing.
                        We set missing positions to NaN before imputation.
      - train_idx/test_idx: indices into df_wide
      - y_scaler (optional): sklearn StandardScaler fitted on TRAIN Y (original scale),
                             used to inverse-transform y_pred/y_true to compute original-scale MSE/RMSE.

    Returns:
      (mse_std, mse_orig, rmse_orig)
        - mse_std: MSE on standardized Y scale
        - mse_orig/rmse_orig: original-scale metrics if y_scaler is provided, else None
    """

    feature_cols = [col for col in df_wide.columns if col.startswith('x')]
    M = len(feature_cols)

    # -----------------------------
    # 1) Parse wide df -> 3D array: (N, T, M)
    # -----------------------------
    def extract_3d_array(df):
        n_samples = len(df)
        sample_seq = df.iloc[0][feature_cols[0]]
        T_local = len(sample_seq) if isinstance(sample_seq, (list, np.ndarray)) else 1

        data_3d = np.zeros((n_samples, T_local, M), dtype=np.float32)
        for i in range(n_samples):
            for j, col in enumerate(feature_cols):
                seq = df.iloc[i][col]
                if isinstance(seq, (list, np.ndarray)):
                    length = min(len(seq), T_local)
                    data_3d[i, :length, j] = np.asarray(seq[:length], dtype=np.float32)
                else:
                    # scalar fallback
                    data_3d[i, 0, j] = float(seq)
        return data_3d, T_local

    data_3d_all, T = extract_3d_array(df_wide)
    y_all = df_wide["Y"].astype(float).values  # likely standardized already

    # -----------------------------
    # 2) Apply mask: set missing X to NaN
    # -----------------------------
    data_3d_masked = data_3d_all.copy()

    if data_mask_full is not None:
        dm = np.asarray(data_mask_full).astype(bool)  # (N, T, M), True = observed
        if dm.shape != data_3d_masked.shape:
            raise ValueError(
                f"data_mask_full shape {dm.shape} does not match X shape {data_3d_masked.shape}. "
                f"Expected (N,T,M) = {data_3d_masked.shape}."
            )
        data_3d_masked[~dm] = np.nan
        actual_missing_rate = np.isnan(data_3d_masked).mean()
        print(f"    [DEBUG] Actual Data Missing Rate: {actual_missing_rate:.1%}")

    # -----------------------------
    # 3) Split Train/Test
    # -----------------------------
    X_train_3d = data_3d_masked[train_idx]
    X_test_3d = data_3d_masked[test_idx]
    y_train = y_all[train_idx]
    y_test = y_all[test_idx]

    y_train_valid_mask = ~np.isnan(y_train)
    y_test_valid_mask = ~np.isnan(y_test)

    # If you already filtered Y-missing before calling, these should be all True.
    if np.sum(~y_train_valid_mask) > 0:
        print(f"    [Y Missing] Train has {np.sum(~y_train_valid_mask)} NaNs -> removing from training")
    if np.sum(~y_test_valid_mask) > 0:
        print(f"    [Y Missing] Test has {np.sum(~y_test_valid_mask)} NaNs -> excluding from evaluation")

    # -----------------------------
    # 4) Reshape to long format: (N*T, M)
    # -----------------------------
    N_train = len(train_idx)
    N_test = len(test_idx)

    X_train_long = X_train_3d.reshape(N_train * T, M)
    X_test_long = X_test_3d.reshape(N_test * T, M)

    # -----------------------------
    # 5) Multiple Imputation loop
    # -----------------------------
    preds = []
    used_fallback = False

    # Guard: MI must be >=1
    n_imputations = int(max(1, n_imputations))

    try:
        for k in range(n_imputations):
            imp_seed = int(random_state + 1000 * k)

            imputer = IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=int(imputer_max_iter),
                sample_posterior=True,
                random_state=imp_seed,
                skip_complete=True,
                initial_strategy='mean',
                verbose=0
            )

            imputer.fit(X_train_long)
            X_train_imp_long = imputer.transform(X_train_long)
            X_test_imp_long = imputer.transform(X_test_long)

            # Flatten to 2D per subject: (N, T*M)
            X_train_imp_flat = X_train_imp_long.reshape(N_train, -1)
            X_test_imp_flat = X_test_imp_long.reshape(N_test, -1)

            # Use only valid Y in training
            X_train_final = X_train_imp_flat[y_train_valid_mask]
            y_train_final = y_train[y_train_valid_mask]

            model = ElasticNet(
                l1_ratio=float(l1_ratio),
                alpha=float(alpha),
                max_iter=2000,
                random_state=int(random_state),
            )
            model.fit(X_train_final, y_train_final)

            preds.append(model.predict(X_test_imp_flat))

    except Exception as e:
        used_fallback = True
        print(f"    [Fallback] IterativeImputer failed ({e}), using Mean Imputation.")

        imputer = SimpleImputer(strategy='mean')
        X_train_imp_long = imputer.fit_transform(X_train_long)
        X_test_imp_long = imputer.transform(X_test_long)

        X_train_imp_flat = X_train_imp_long.reshape(N_train, -1)
        X_test_imp_flat = X_test_imp_long.reshape(N_test, -1)

        X_train_final = X_train_imp_flat[y_train_valid_mask]
        y_train_final = y_train[y_train_valid_mask]

        # fallback alpha slightly smaller (can learn more)
        model = ElasticNet(
            l1_ratio=float(l1_ratio),
            alpha=0.01,
            max_iter=2000,
            random_state=int(random_state),
        )
        model.fit(X_train_final, y_train_final)
        preds.append(model.predict(X_test_imp_flat))

    # -----------------------------
    # 6) Evaluate (standardized scale first)
    # -----------------------------
    y_pred_all = np.mean(np.stack(preds, axis=0), axis=0)  # (N_test,)
    y_pred_valid = y_pred_all[y_test_valid_mask]
    y_test_valid = y_test[y_test_valid_mask]

    mse_std = float(mean_squared_error(y_test_valid, y_pred_valid))
    print(f"    [Result] MSE(std): {mse_std:.6f} | n_imputations={n_imputations} | fallback={used_fallback}")

    # -----------------------------
    # 7) Original-scale metrics (optional)
    # -----------------------------
    if y_scaler is None:
        mse_orig = None
        rmse_orig = None
    else:
        y_pred_orig = y_scaler.inverse_transform(y_pred_valid.reshape(-1, 1)).ravel()
        y_true_orig = y_scaler.inverse_transform(y_test_valid.reshape(-1, 1)).ravel()
        mse_orig = float(mean_squared_error(y_true_orig, y_pred_orig))
        rmse_orig = float(np.sqrt(mse_orig))
        print(f"    [Result] MSE(orig): {mse_orig:.6f} | RMSE(orig): {rmse_orig:.6f}")

    return mse_std, mse_orig, rmse_orig
