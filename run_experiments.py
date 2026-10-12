"""Run the simulation study.

Examples
--------
    python run_experiments.py                                   # complete data + 3 missingness mechanisms
    python run_experiments.py --experiments MNAR --n-runs 10    # a single mechanism
    python run_experiments.py --models Proposed Classic --M 50  # subset of models, smaller problem

Results go to ``<out-dir>/<experiment>/``: per-run MSE tables, ``summary.json`` (updated after every
repetition) and ``checkpoints.txt`` (mean/std appended every ``--checkpoint-every`` repetitions).
"""
import argparse
import os

from temporal_encoding.models import ALL_MODELS
from temporal_encoding.runner import run_simulations

EXPERIMENTS = {
    "complete": "No missing data (baseline).",
    "MCAR": "Missing completely at random: P(missing) is constant.",
    "MAR_causal": "MAR: P(missing | observed causal covariates).",
    "MAR_noncausal": "MAR: P(missing | observed non-causal covariates).",
    "MNAR": "MNAR: P(missing | the unobserved value itself).",
}

DEFAULT_MODELS = ["Proposed", "Classic", "Shortformer", "SPE", "Spacetimeformer", "ElasticNet"]
DEFAULT_EXPERIMENTS = ["complete", "MAR_causal", "MAR_noncausal", "MNAR"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, choices=list(EXPERIMENTS))
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=ALL_MODELS)
    p.add_argument("--n-runs", type=int, default=30, help="Monte-Carlo repetitions per experiment.")
    p.add_argument("--T", type=int, default=20, help="Number of follow-up visits (T+1 time points are generated).")
    p.add_argument("--N", type=int, default=200, help="Subjects before outcome-missingness filtering.")
    p.add_argument("--M", type=int, default=200, help="Number of longitudinal features.")
    p.add_argument("--rates", nargs="+", type=float, default=[0.1, 0.3, 0.5], help="X-missing rates.")
    p.add_argument("--missing-rate-y", type=float, default=0.1)
    p.add_argument("--memory-lags", type=int, default=5, help="Autoregressive lags in the covariate process.")
    p.add_argument("--memory-tau", type=float, default=3.0, help="Decay of the autoregressive weights.")
    p.add_argument("--batch-size", type=int, default=40)
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--d-model", type=int, default=36)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--out-dir", default="outputs")
    return p.parse_args()


def main():
    args = parse_args()
    for exp in args.experiments:
        print(f"\n{'=' * 80}\n  {exp}: {EXPERIMENTS[exp]}\n{'=' * 80}")
        run_simulations(
            T=args.T, N=args.N, M=args.M,
            model_list=args.models,
            n_runs=args.n_runs,
            missing_mechanism=None if exp == "complete" else exp,
            missing_rates_x=args.rates,
            missing_rate_y=args.missing_rate_y,
            memory_lags=args.memory_lags,
            memory_tau=args.memory_tau,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            d_model=args.d_model,
            num_layers=args.num_layers,
            out_dir=os.path.join(args.out_dir, exp),
            checkpoint_every=args.checkpoint_every,
        )


if __name__ == "__main__":
    main()
