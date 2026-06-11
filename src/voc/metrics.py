"""Paper evaluation metrics (rffexhibits_function.m semantics).

All metrics are computed per seed and per (P, lambda) config on MONTHLY
values; sqrt(12) annualization is a reporting concern and never happens
here. The OLS of the timing strategy on the market answers a different
question than the Sharpe ratio: alpha (and IR = alpha / residual std)
measures value added BEYOND static market exposure — what an allocator
would look at.

Everything is closed-form and vectorized over the [K, nP, nL] prediction
cube: fitting nP*nL separate OLS regressions through statsmodels would be
thousands of Python-loop iterations for what is algebraically two moments
and a ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from voc.constants import PERCENTILES


def seed_config_metrics(
    yprd: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    *,
    bnrm: np.ndarray | None = None,
    subsample: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    """Metrics for one seed, [nP, nL] per key.

    yprd: [K, nP, nL] OOS forecasts; y: [K] realized (vol-scaled) returns;
    dates: [K] yyyymm ints; bnrm: optional [K, nP, nL] squared beta norms
    (MATLAB Bnrm), reported as their time average per config.
    subsample = (first_year, last_year) inclusive.
    """
    if subsample is not None:
        beg, end = subsample
        mask = (dates >= beg * 100 + 1) & (dates <= end * 100 + 12)
        yprd, y = yprd[mask], y[mask]
        if bnrm is not None:
            bnrm = bnrm[mask]

    n = y.shape[0]
    y_col = y[:, None, None]

    # R2 vs the zero forecast: 1 - var(yprd - y) / var(y), ddof=1.
    err = yprd - y_col
    r2 = 1.0 - err.var(axis=0, ddof=1) / y.var(ddof=1)

    # Timing strategy: position = forecast, return = forecast * realized.
    # Degenerate configs (constant zero forecast) have vol = 0; their
    # ratios are NaN by design, matching MATLAB.
    timing = yprd * y_col
    er = timing.mean(axis=0)
    vol = timing.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = er / vol

    # OLS timing ~ a + b*market, closed form. Centered cross-moments give
    # the slope; residual moments give the t-stat exactly as statsmodels
    # (s^2 = RSS/(n-2), se(a)^2 = s^2 * (1/n + ybar^2/Sxx)).
    y_mean = y.mean()
    y_center = y - y_mean
    sxx = float(y_center @ y_center)
    slope = np.einsum("kpl,k->pl", timing - er, y_center) / sxx
    alpha = er - slope * y_mean

    resid = timing - alpha - slope * y_col
    rss = np.einsum("kpl,kpl->pl", resid, resid)
    s2 = rss / (n - 2)
    alpha_se = np.sqrt(s2 * (1.0 / n + y_mean**2 / sxx))

    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "r2": r2,
            "er": er,
            "vol": vol,
            "sr": sr,
            "alpha": alpha,
            "alpha_t": alpha / alpha_se,
            "ir": alpha / np.sqrt(rss / (n - 1)),
        }
    if bnrm is not None:
        out["bnrm"] = bnrm.mean(axis=0)
    return out


def _pct_label(q: float) -> str:
    return f"p{q}".replace(".", "_")


def aggregate_seed_metrics(
    per_seed: list[dict[str, np.ndarray]],
    *,
    p_grid: list[int],
    lambdas: list[float],
) -> pd.DataFrame:
    """Cross-seed mean and percentile bands, long format.

    One row per (P, lambda); columns <metric>_mean and <metric>_p<q> for
    the paper's percentile list. The mean curve is what the paper plots;
    the bands show Monte Carlo dispersion across weight draws.
    """
    metric_names = list(per_seed[0].keys())
    stacked = {name: np.stack([m[name] for m in per_seed]) for name in metric_names}  # [S, nP, nL]

    rows = []
    for pi, p in enumerate(p_grid):
        for li, lam in enumerate(lambdas):
            row: dict[str, float | int] = {"p": p, "lam": lam, "n_seeds": len(per_seed)}
            for name in metric_names:
                values = stacked[name][:, pi, li]
                row[f"{name}_mean"] = float(np.nanmean(values))
                for q, pct in zip(PERCENTILES, np.nanpercentile(values, PERCENTILES), strict=True):
                    row[f"{name}_{_pct_label(q)}"] = float(pct)
            rows.append(row)
    return pd.DataFrame(rows)
