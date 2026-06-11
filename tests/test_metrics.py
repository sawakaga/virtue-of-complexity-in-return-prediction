"""Tests for the paper's evaluation metrics (rffexhibits_function.m).

Per seed and per (P, lambda) config:
- R2 = 1 - var(forecast error) / var(realized)   [variance-based, ddof=1]
- timing strategy return = forecast * realized
- ER/vol/SR monthly; sqrt(12) annualization happens only at reporting
- alpha, alpha t-stat, IR = alpha/std(resid) from OLS timing ~ 1 + market
Cross-seed: mean and the paper's percentile bands.
"""

import numpy as np
import statsmodels.api as sm

from voc.metrics import aggregate_seed_metrics, seed_config_metrics

RNG = np.random.default_rng(13)


def _toy_inputs(k=60, n_p=2, n_l=3):
    y = RNG.normal(size=k)
    yprd = RNG.normal(size=(k, n_p, n_l))
    dates = np.array([193001 + 100 * (i // 12) + (i % 12) for i in range(k)], dtype=np.int64)
    return yprd, y, dates


def test_r2_is_one_for_perfect_forecast_and_zero_for_zero_forecast():
    yprd, y, dates = _toy_inputs()
    yprd[:, 0, 0] = y  # perfect
    yprd[:, 0, 1] = 0.0  # zero forecast

    m = seed_config_metrics(yprd, y, dates)

    np.testing.assert_allclose(m["r2"][0, 0], 1.0)
    # 1 - var(0 - y)/var(y) = 0: the zero forecast is the R2 benchmark.
    np.testing.assert_allclose(m["r2"][0, 1], 0.0, atol=1e-12)


def test_timing_moments_hand_computed():
    yprd, y, dates = _toy_inputs()
    m = seed_config_metrics(yprd, y, dates)

    timing = yprd[:, 1, 2] * y
    np.testing.assert_allclose(m["er"][1, 2], timing.mean(), rtol=1e-12)
    np.testing.assert_allclose(m["vol"][1, 2], timing.std(ddof=1), rtol=1e-12)
    np.testing.assert_allclose(m["sr"][1, 2], timing.mean() / timing.std(ddof=1), rtol=1e-12)


def test_alpha_ir_match_statsmodels_ols():
    yprd, y, dates = _toy_inputs()
    m = seed_config_metrics(yprd, y, dates)

    timing = yprd[:, 0, 1] * y
    ols = sm.OLS(timing, sm.add_constant(y)).fit()
    alpha = ols.params[0]

    np.testing.assert_allclose(m["alpha"][0, 1], alpha, rtol=1e-10)
    np.testing.assert_allclose(m["alpha_t"][0, 1], ols.tvalues[0], rtol=1e-10)
    np.testing.assert_allclose(m["ir"][0, 1], alpha / ols.resid.std(ddof=1), rtol=1e-10)


def test_subsample_bounds_filter_by_year():
    yprd, y, dates = _toy_inputs(k=48)
    full = seed_config_metrics(yprd, y, dates)
    sub = seed_config_metrics(yprd, y, dates, subsample=(1930, 1931))

    mask = (dates >= 193001) & (dates <= 193112)
    timing = yprd[mask, 0, 0] * y[mask]
    np.testing.assert_allclose(sub["er"][0, 0], timing.mean(), rtol=1e-12)
    assert not np.allclose(full["er"][0, 0], sub["er"][0, 0])


def test_aggregate_mean_and_percentiles_across_seeds():
    yprd, y, dates = _toy_inputs()
    per_seed = [seed_config_metrics(yprd + 0.1 * s, y, dates) for s in range(5)]

    agg = aggregate_seed_metrics(per_seed, p_grid=[8, 24], lambdas=[0.1, 1.0, 10.0])

    row = agg[(agg["p"] == 8) & (agg["lam"] == 0.1)].iloc[0]
    stacked = np.array([m["sr"][0, 0] for m in per_seed])
    np.testing.assert_allclose(row["sr_mean"], stacked.mean(), rtol=1e-12)
    np.testing.assert_allclose(row["sr_p50"], np.percentile(stacked, 50), rtol=1e-12)
    np.testing.assert_allclose(row["sr_p2_5"], np.percentile(stacked, 2.5), rtol=1e-12)
    assert row["n_seeds"] == 5
    assert len(agg) == 6  # 2 grid points x 3 lambdas
