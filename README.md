# The Virtue of Complexity in Return Prediction — Python Reproduction

Reproduction of Kelly, Malamud, Zhou, *"The Virtue of Complexity in Return
Prediction"* (The Journal of Finance, 2023): out-of-sample US market timing
with Random Fourier Features + rolling ridge regression, swept across model
complexity c = P/T.

The implementation is verified against the authors' MATLAB code (held
outside this repo) at three levels: the constructed predictors match
`GYdata.mat` column-by-column, a float64 NumPy oracle transliterates the
MATLAB fit loop, and the fast torch solver is parity-tested against that
oracle.

## Method (matching the MATLAB reference)

1. **Data** — 14 constructed Goyal-Welch predictors (dfy, infl, svar, de,
   lty, tms, tbl, dfr, dp, dy, ltr, ep, bm, ntis) + the lagged market
   excess return (CRSP_SPvw − Rfree), all lagged one month.
2. **Standardization** — predictors divided by an expanding std (no
   demeaning); target divided by the RMS of the previous 12 monthly
   returns (strictly lagged — no look-ahead); first 36 months dropped.
   Usable sample: 1930-01 to 2024-12.
3. **Features** — `Z = [cos(γ·W·x); sin(γ·W·x)]`, `W ~ N(0, I)`, γ=2,
   drawn once at maxP=12000 and prefix-sliced over the paper's non-uniform
   P grid (dense near the interpolation threshold P = T).
4. **Fit** — rolling window T ∈ {12, 60, 120}: train on months
   [k, k+T), re-standardize features by the training window's std, ridge
   with effective penalty λ·T for λ ∈ 10^{−3..3}, predict month k+T.
5. **Evaluation** — per seed: R² = 1 − var(err)/var(y), timing strategy
   return = forecast × realized, monthly Sharpe, OLS alpha / IR vs the
   market; then mean + percentile bands across seeds.

## Usage

```bash
uv sync --extra dev

# Full run: all three windows, 10 seeds, auto device (CUDA > MPS > CPU)
uv run python -m voc.cli

# Custom: one window, more seeds, raw prediction dump
uv run python -m voc.cli --windows 120 --n-seeds 100 --save-predictions

# Smoke run (seconds)
uv run python -m voc.cli --windows 12 --n-seeds 2 --max-p 120
```

Outputs land in `artifacts/metrics_T<window>.parquet` (long format: one
row per subsample × P × λ, columns `<metric>_mean` and percentile bands;
all values **monthly** — annualize at reporting). Figures:

```python
import pandas as pd
from pathlib import Path
from voc.plots import plot_window_metrics

frame = pd.read_parquet("artifacts/metrics_T120.parquet")
plot_window_metrics(frame, out_dir=Path("artifacts/figures"))
```

Development: `make test`, `make lint`, `make bench` (cross-device solver
benchmark).

## Engineering notes

- The solver works entirely in T-space (dual/representer form): the
  within-window standardized Gram is block-additive over features, so the
  P grid is swept incrementally — each feature's matmul happens exactly
  once. One batched `eigh` per grid point serves all 7 λ values
  (`K + λT·I` shares eigenvectors). β is never materialized; ‖β‖² comes
  from the eigen-coefficients.
- fp64 on CPU/CUDA by default. MPS runs fp32 (no fp64 hardware) with
  `eigh` falling back to CPU (Metal lacks it).
- A slow float64 oracle (`voc/reference.py`) exists only for tests.

## Known deviations from the paper

| Deviation | Reason |
|---|---|
| RNG stream differs from MATLAB `rng(s); randn` | Not bit-reproducible from NumPy/torch; parity is established by injecting identical W in tests. Cross-seed averages are the comparable object. |
| `bm` differs after 2009-03, `ntis` slightly | Goyal-Welch restated values between data vintages (we use the 2024 vintage; GYdata.mat is ~2021). Pre-revision parity is exact. |
| Sample extends to 2024-12 | GYdata stops 2020-12; subsample metrics (1926–2020 etc.) remain comparable. |
| Default 10 seeds vs the paper's 1000 | Compute budget; `--n-seeds 1000` reproduces the protocol on a bigger machine. |

## Layout

```
src/voc/          package (data, preprocess, grids, rff, solver, metrics, plots, device, cli)
tests/            pytest suite incl. GYdata.mat parity fixtures
data/             Goyal-Welch raw data (decimal-comma CSV) + Fama-French factors
scripts/bench.py  cross-device benchmark
archive/          earlier exploratory implementations
```
