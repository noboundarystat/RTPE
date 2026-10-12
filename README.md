# Temporal encoding for transformers on longitudinal data with missingness

Simulation study comparing a transformer that encodes the **real (continuous) measurement time** with a
regular transformer (visit-index positional encoding), Shortformer, SPE, Spacetimeformer and an ElasticNet
baseline. The task is to predict a scalar outcome from irregularly spaced longitudinal covariates when
covariate values are missing under MAR (causal / non-causal covariates) or MNAR mechanisms at increasing
missing rates.

## Models

All neural models share one skeleton (`temporal_encoding/models/common.py`), so differences in test error
come from how time / position enters the model:

```
value -> embedding (+ time / position information)
      -> transformer encoder shared across features, run per (subject, feature) over the visits
      -> masked mean pooling over observed visits
      -> concatenate features -> fusion MLP -> cross-feature MLP -> linear regression head
```

| Name in code | File | Time / position handling |
|---|---|---|
| `Proposed` | `models/proposed.py` | Sinusoidal encoding of the **real measurement time**, added at observed visits only |
| `Classic` | `models/classic.py` | Standard sinusoidal encoding of the **visit index** |
| `Shortformer` | `models/shortformer.py` | Position-infused attention: positions enter Q and K of every layer (Press et al., 2021) |
| `SPE` | `models/spe.py` | Stochastic positional encoding, sine variant with gating (Liutkus et al., 2021) |
| `Spacetimeformer` | `models/spacetimeformer.py` | Value + Time2Vec projection, learned position and variable embeddings (Grigsby et al., 2021; Kazemi et al., 2019) |
| `ElasticNet` | `models/elasticnet.py` | ElasticNet on the flattened time x feature matrix after `IterativeImputer` (mean imputation if it fails) |

Missing values in the neural models: a missing cell is zeroed and excluded from attention and from the
pooling average; a (subject, feature) trajectory with no observed visit contributes a zero vector.

> Earlier result files in `results/` use the previous names `TransformerV2` (= `Proposed`) and
> `ClassicTransformerV2` (= `Classic`).

## Simulation design (`temporal_encoding/simulation.py`)

* **Time grid**: `T+1` visits with growing gaps: daily, then weekly, then 30-day steps (`T=20` gives 21 visits,
  days 0 to 288). The grid is shared by all subjects, so irregular spacing, not subject-specific schedules, is
  what separates real-time from index-based encodings.
* **Covariates**: `M` features, equicorrelated within a visit (rho = 0.5), autoregressive over time with
  `--memory-lags` lags and exponentially decaying weights (`--memory-tau`, total weight 0.8), then a
  log transform to make them skewed.
* **Outcome**: a sum of `sin(pi * x)` over 10 randomly chosen (visit, feature) cells with U(-1, 1)
  coefficients, plus noise (noise / total variance = 0.1). Most features are therefore pure noise.
* **Missingness** (nested: a smaller rate's missing set is contained in a larger rate's):
  * `MCAR`: independent Bernoulli.
  * `MAR_causal` / `MAR_noncausal`: P(missing) is a logistic function of the (observed) sum of the causal /
    non-causal covariates at that visit; a missing visit is missing for all features.
  * `MNAR`: only the causal features can be missing, with probability depending on their own value.
  * The intercept of each logistic model is solved numerically to hit the requested rate. Subjects with a
    missing outcome (target 10%, mechanism-dependent) are dropped.
* **Protocol**: subjects are split 40/30/30 into train/validation/test (same split for all models and
  rates); Y is standardized with training statistics and each feature with its observed training values.
  Adam, MSE loss, gradient clipping at 5, batch size 40, at most 50 epochs, early stopping on validation
  MSE (patience / minimum epochs depend on the missing rate, see `runner.py`). `d_model=36`, 2 layers, 2 heads,
  dropout 0.2. Repetition `r` uses seed `r`, so all models see identical data. Reported metric: test MSE on the
  standardized outcome (also original-scale MSE / RMSE).

## Installation

```bash
pip install -r requirements.txt
python -m pytest -q tests        # smoke tests (a few seconds)
```

Developed with Python 3.12 and PyTorch 2.13 (CPU).

## Running experiments

```bash
# complete data + MAR_causal + MAR_noncausal + MNAR, 30 repetitions each, rates 0.1 / 0.3 / 0.5
python run_experiments.py

# one mechanism, fewer repetitions
python run_experiments.py --experiments MNAR --n-runs 10

# subset of models and a smaller problem for a quick check
python run_experiments.py --models Proposed Classic ElasticNet --T 10 --N 100 --M 20 --n-runs 3
```

`python run_experiments.py --help` lists all options (`MCAR` is available but not run by default).
`scripts/submit_lsf.sh` is an example LSF batch script.

Runtime is dominated by the neural models: with the default `N=M=200` one repetition of the complete-data
experiment took about 20 minutes on a laptop CPU, and the missing-data experiments train every model once per
missing rate.

### Output

Written to `outputs/<experiment>/` (git-ignored) **incrementally**, so stopping a run loses nothing that has
finished:

* `per_run_mse_std_<model>.csv`: test MSE of every finished repetition, one column per missing rate
  (rewritten after every repetition).
* `summary.json`: mean, SD (ddof = 0) and count per model, rate and metric (`mse_std`, `mse_orig`,
  `rmse_orig`); updated after every repetition.
* `checkpoints.txt`: the same summary appended every `--checkpoint-every` repetitions (default 10).

## Layout

```
run_experiments.py          command-line entry point
temporal_encoding/
  simulation.py             data generation and missingness mechanisms
  dataset.py                PyTorch dataset (standardization, padding, masks)
  training.py               training loop with validation early stopping
  runner.py                 repetitions, splits, per-rate fitting, output files
  models/                   shared skeleton + one file per model
tests/                      smoke tests
scripts/                    example cluster job script
results/                    preliminary results (complete data, 30 repetitions)
```

## Implementation notes

* The per-feature loop is vectorised by folding the feature axis into the batch axis; this is exactly
  equivalent to looping over features because attention never mixes across the batch axis.
* The regression head has no output activation. An earlier version ended in a GELU, which cannot output
  values much below -0.17 and so is unsuitable for standardized targets.
* A (subject, feature) trajectory with no observed visit would give an all-masked attention row (NaN); such
  rows fall back to the valid visits inside the encoder and their pooled output is set to zero.
* SPE draws new random realizations on every forward pass, including at evaluation, so its predictions are
  stochastic by design.
* Runs are seeded and use deterministic PyTorch algorithms; results are reproducible on the same
  software/hardware.

## References

* Press, Smith, Lewis (2021). *Shortformer: Better Language Modeling using Shorter Inputs.*
* Liutkus et al. (2021). *Relative Positional Encoding for Transformers with Linear Complexity.*
* Grigsby, Wang, Qi (2021). *Long-Range Transformers for Dynamic Spatiotemporal Forecasting.*
* Kazemi et al. (2019). *Time2Vec: Learning a Vector Representation of Time.*
