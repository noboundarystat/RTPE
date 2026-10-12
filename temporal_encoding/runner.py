"""Monte-Carlo simulation runner.

For every repetition (seed = run index):
  1. simulate covariates, outcome and missingness masks;
  2. drop subjects whose outcome is missing;
  3. split subjects 60/20/20 into train/validation/test (identical for all models and rates);
  4. standardize Y with training statistics;
  5. for each X-missing rate: refit feature scalers on observed training values, then train every
     model from scratch and record test MSE.

Results are written incrementally (per-run tables after every repetition, a summary every
``checkpoint_every`` repetitions) so a stopped run keeps everything finished so far.
"""
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from .dataset import LongitudinalDataset
from .models import ELASTICNET, TORCH_MODELS, train_and_evaluate_elasticnet
from .simulation import simulate_data
from .training import predict, train_once

METRICS = ("mse_std", "mse_orig", "rmse_orig")

# Optimizer / early-stopping settings. They depend only on the missing rate, never on the model.
COMPLETE_CONFIG = dict(lr=0.002, patience=5, min_epochs=15, improve_tol=1e-4)
MISSING_CONFIG = {
    0.1: dict(lr=0.001, patience=5, min_epochs=15, improve_tol=1e-4),
    0.3: dict(lr=0.001, patience=4, min_epochs=12, improve_tol=2e-4),
    0.5: dict(lr=0.001, patience=3, min_epochs=10, improve_tol=3e-4),
}
DEFAULT_MISSING_CONFIG = dict(lr=0.001, patience=5, min_epochs=5, improve_tol=1e-3)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic_backend():
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _take(df, idx):
    return df.iloc[idx].reset_index(drop=True)


def _get_rate_mask(data_masks, rate):
    for k, v in data_masks.items():
        if np.isclose(float(k), float(rate)):
            return v
    raise KeyError(f"Missing rate {rate} not in data_masks (keys: {list(data_masks)})")


def _scores(preds, labels, y_scaler):
    """MSE on the standardized scale and MSE / RMSE on the original outcome scale."""
    mse_std = float(((preds - labels) ** 2).mean())
    y_pred = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(labels.reshape(-1, 1)).ravel()
    mse_orig = float(((y_pred - y_true) ** 2).mean())
    return mse_std, mse_orig, float(np.sqrt(mse_orig))


def _rate_label(tag, rate):
    return "Complete" if tag == "complete" else f"{tag} {round(rate * 100)}%"


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------
def _summarize(results, model_list, rates):
    return {
        metric: {
            m: {
                f"{r:g}": {
                    "mean": float(np.mean(results[metric][m][r])) if results[metric][m][r] else None,
                    "std": float(np.std(results[metric][m][r])) if results[metric][m][r] else None,
                    "n": len(results[metric][m][r]),
                }
                for r in rates
            }
            for m in model_list
        }
        for metric in METRICS
    }


def _write_outputs(out_dir, tag, model_list, rates, results, n_done, checkpoint):
    os.makedirs(out_dir, exist_ok=True)

    for m in model_list:
        table = {_rate_label(tag, r): results["mse_std"][m][r] for r in rates}
        pd.DataFrame(table).to_csv(os.path.join(out_dir, f"per_run_mse_std_{m}.csv"), index=False)

    summary = _summarize(results, model_list, rates)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"experiment": tag, "n_runs_completed": n_done, "metrics": summary}, f, indent=2)

    if checkpoint:
        with open(os.path.join(out_dir, "checkpoints.txt"), "a") as f:
            f.write(f"\n===== After {n_done} simulations =====\n")
            for m in model_list:
                for r in rates:
                    s = {k: summary[k][m][f"{r:g}"] for k in METRICS}
                    f.write(
                        f"{m} | rate={r:g}: "
                        f"Mean MSE(std)={s['mse_std']['mean']:.3f}, Std={s['mse_std']['std']:.3f} | "
                        f"Mean MSE(orig)={s['mse_orig']['mean']:.3f}, Std={s['mse_orig']['std']:.3f} | "
                        f"Mean RMSE(orig)={s['rmse_orig']['mean']:.3f}, Std={s['rmse_orig']['std']:.3f}\n"
                    )
        print(f"Checkpoint after {n_done} simulations written to {out_dir}")


# --------------------------------------------------------------------------------------
# One repetition
# --------------------------------------------------------------------------------------
def _simulate_run(run, T, N, M, missing_mechanism, rates, missing_rate_y, transform,
                  memory_lags, memory_tau):
    """Simulate data for one repetition and return the split, scalers and masks."""
    set_all_seeds(run)
    set_deterministic_backend()

    kwargs = dict(seed=run, T=T, N=N, M=M, transform=transform,
                  memory_lags=memory_lags, memory_tau=memory_tau)
    if missing_mechanism is None:
        df_wide, df_time, Y_true, _, _ = simulate_data(**kwargs)
        data_masks = None
    else:
        df_wide, df_time, Y_true, _, _, data_masks, Y_mask = simulate_data(
            **kwargs, missing_mechanism=missing_mechanism,
            missing_rates_x=list(rates), missing_rate_y=missing_rate_y,
        )

    df_wide = df_wide.copy()
    df_time = df_time.copy()
    df_wide["Y"] = Y_true.astype(float)

    # Subjects with a missing outcome are dropped entirely.
    if data_masks is not None:
        observed = np.where(np.asarray(Y_mask).astype(bool))[0]
        df_wide = df_wide.iloc[observed].reset_index(drop=True)
        df_time = df_time.iloc[observed].reset_index(drop=True)
        data_masks = {
            r: np.asarray(_get_rate_mask(data_masks, r)).astype(bool)[:, observed, :] for r in rates
        }

    all_idx = list(range(len(df_wide)))
    train_idx, temp_idx = train_test_split(all_idx, test_size=0.40, random_state=run)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=run)

    y_scaler = StandardScaler().fit(df_wide.iloc[train_idx]["Y"].to_numpy().reshape(-1, 1))
    df_wide["Y"] = y_scaler.transform(df_wide["Y"].to_numpy().reshape(-1, 1)).ravel()

    return df_wide, df_time, data_masks, (train_idx, val_idx, test_idx), y_scaler


def _make_loaders(df_wide, df_time, splits, mask_ntm, time_length, batch_size, run):
    """Datasets/loaders for one missing rate; feature scalers use observed training values only."""
    train_idx, val_idx, test_idx = splits
    parts = {}
    scalers = None
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        parts[name] = LongitudinalDataset(
            _take(df_wide, idx), _take(df_time, idx), max_len=time_length,
            data_mask=None if mask_ntm is None else mask_ntm[idx],
            scalers=scalers, fit_scalers=(name == "train"),
        )
        scalers = parts["train"].scalers

    generator = torch.Generator().manual_seed(run)
    return (
        DataLoader(parts["train"], batch_size=batch_size, shuffle=True, generator=generator),
        DataLoader(parts["val"], batch_size=batch_size, shuffle=False),
        DataLoader(parts["test"], batch_size=batch_size, shuffle=False),
    )


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def run_simulations(
    T, N, M, model_list, n_runs,
    missing_mechanism=None,
    missing_rates_x=(0.1, 0.3, 0.5),
    missing_rate_y=0.1,
    transform="lognormal",
    memory_lags=5,
    memory_tau=3.0,
    batch_size=40,
    num_epochs=50,
    d_model=36,
    num_layers=2,
    out_dir=None,
    checkpoint_every=10,
):
    """Run ``n_runs`` repetitions and return ``{metric: {model: {rate: [per-run values]}}}``.

    ``missing_mechanism=None`` runs the complete-data experiment (single "rate" 0.0).
    Valid mechanisms: ``MCAR``, ``MAR_causal``, ``MAR_noncausal``, ``MNAR``.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    has_missing = missing_mechanism is not None
    rates = sorted(float(r) for r in missing_rates_x) if has_missing else [0.0]
    tag = missing_mechanism if has_missing else "complete"

    print(f"\n[Config] experiment={tag} T={T} N={N} M={M} n_runs={n_runs} batch_size={batch_size} "
          f"memory_lags={memory_lags} memory_tau={memory_tau} num_epochs={num_epochs} "
          f"rates={rates} models={list(model_list)}")

    results = {metric: {m: {r: [] for r in rates} for m in model_list} for metric in METRICS}

    for run in range(n_runs):
        print(f"\n===== Simulation {run + 1}/{n_runs} =====")
        df_wide, df_time, data_masks, splits, y_scaler = _simulate_run(
            run, T, N, M, missing_mechanism, rates, missing_rate_y, transform,
            memory_lags, memory_tau,
        )
        time_length = len(df_time.iloc[0]["time"])

        for rate in rates:
            if has_missing:
                cfg = MISSING_CONFIG.get(round(rate, 4), DEFAULT_MISSING_CONFIG)
                mask_ntm = np.transpose(_get_rate_mask(data_masks, rate), (1, 0, 2)).astype(bool)
                print(f"\n--- X missing rate = {rate} | {cfg} ---")
            else:
                cfg = COMPLETE_CONFIG
                mask_ntm = None

            train_loader, val_loader, test_loader = _make_loaders(
                df_wide, df_time, splits, mask_ntm, time_length, batch_size, run
            )

            for model_name in model_list:
                print(f"\n--- Training model: {model_name} | rate = {rate} ---")

                if model_name == ELASTICNET:
                    scores = train_and_evaluate_elasticnet(
                        df_wide=df_wide, train_idx=splits[0], test_idx=splits[2],
                        data_mask_full=mask_ntm, random_state=run, y_scaler=y_scaler,
                    )
                else:
                    model = TORCH_MODELS[model_name](
                        n_features=M, d_model=d_model, num_layers=num_layers, max_len=time_length
                    ).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
                    train_once(
                        model, train_loader, val_loader, torch.nn.MSELoss(), optimizer,
                        num_epochs, device,
                        patience=cfg["patience"], min_epochs=cfg["min_epochs"],
                        improve_tol=cfg["improve_tol"],
                    )
                    preds, labels = predict(model, test_loader, device)
                    scores = _scores(preds, labels, y_scaler)

                print(f"[TEST] {model_name} | rate={rate} | MSE(std)={scores[0]:.6f} | "
                      f"MSE(orig)={scores[1]:.6f} | RMSE(orig)={scores[2]:.6f}")
                for metric, value in zip(METRICS, scores):
                    results[metric][model_name][rate].append(value)

        if out_dir is not None:
            _write_outputs(out_dir, tag, model_list, rates, results, run + 1,
                           checkpoint=((run + 1) % checkpoint_every == 0))

    print("\n===== Final Results =====")
    summary = _summarize(results, model_list, rates)
    for m in model_list:
        for r in rates:
            s = {k: summary[k][m][f"{r:g}"] for k in METRICS}
            print(f"{m} | rate={r:g}: "
                  f"MSE(std)={s['mse_std']['mean']:.3f} +/- {s['mse_std']['std']:.3f} | "
                  f"MSE(orig)={s['mse_orig']['mean']:.3f} +/- {s['mse_orig']['std']:.3f} | "
                  f"RMSE(orig)={s['rmse_orig']['mean']:.3f} +/- {s['rmse_orig']['std']:.3f}")

    return results
