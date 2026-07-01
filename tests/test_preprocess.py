"""Tests for the standardization pipeline (volstdbwd.m + target vol scaling).

The defining property under test is causality: every scaling statistic must
be computable from information available at decision time. The perturbation
tests prove this mechanically instead of relying on index inspection.
"""

from pathlib import Path

import numpy as np
import pytest

from voc.constants import BURN_IN_MONTHS, GW_PREDICTOR_ORDER
from voc.preprocess import prepare_dataset, target_vol_scale, volstd_expanding

FIXTURE = Path(__file__).parent / "fixtures" / "gydata_slice.npz"

RNG = np.random.default_rng(7)


def test_volstd_first_block_uses_initial_36_row_std():
    x = RNG.normal(size=(50, 3)) * np.array([1.0, 10.0, 0.1])
    out = volstd_expanding(x, min_obs=36)

    expected_std = np.std(x[:36], axis=0, ddof=1)
    np.testing.assert_allclose(out[:36], x[:36] / expected_std, rtol=1e-12)


def test_volstd_row_t_uses_expanding_std_inclusive():
    x = RNG.normal(size=(60, 2))
    out = volstd_expanding(x, min_obs=36)

    # MATLAB: Xout(t,:) = X(t,:) / nanstd(X(1:t,:)) — std includes row t.
    t = 45
    expected_std = np.std(x[: t + 1], axis=0, ddof=1)
    np.testing.assert_allclose(out[t], x[t] / expected_std, rtol=1e-12)


def test_volstd_scales_but_never_demeans():
    # A series with a large positive mean must keep every value positive:
    # volstdbwd divides by std only, it never subtracts the mean.
    x = RNG.normal(size=(80, 1)) + 100.0
    out = volstd_expanding(x, min_obs=36)
    assert (out > 0).all()


def test_volstd_uses_ddof_1_like_matlab_nanstd():
    # ddof=0 vs ddof=1 differ by sqrt(n/(n-1)); with n=36 that is ~1.4%.
    x = RNG.normal(size=(36, 1))
    out = volstd_expanding(x, min_obs=36)
    np.testing.assert_allclose(out[:36, 0], x[:, 0] / x[:, 0].std(ddof=1), rtol=1e-12)


def test_volstd_ignores_nans_in_std():
    x = RNG.normal(size=(40, 1))
    x[5, 0] = np.nan
    out = volstd_expanding(x, min_obs=36)
    valid = np.delete(x[:36, 0], 5)
    np.testing.assert_allclose(out[0, 0], x[0, 0] / np.std(valid, ddof=1), rtol=1e-12)


def test_target_vol_scale_formula():
    y = RNG.normal(size=30)
    out = target_vol_scale(y, months=12)

    t = 20
    trailing = y[t - 12 : t]  # strictly t-12 .. t-1
    expected = y[t] / np.sqrt(np.mean(trailing**2))
    np.testing.assert_allclose(out[t], expected, rtol=1e-12)

    # The first `months` entries lack a full trailing window.
    assert np.isnan(out[:12]).all()
    assert np.isfinite(out[12:]).all()


def test_target_vol_scale_has_no_look_ahead():
    # Perturbing y[t] must not change the *scaler* applied at t (only the
    # numerator), and must not affect any output before t+1.
    y = RNG.normal(size=40)
    base = target_vol_scale(y, months=12)

    t = 25
    perturbed = y.copy()
    perturbed[t] = 99.0
    out = target_vol_scale(perturbed, months=12)

    # Outputs strictly before t are untouched.
    np.testing.assert_array_equal(out[:t], base[:t])
    # At t the denominator is unchanged: output scales linearly with y[t].
    np.testing.assert_allclose(out[t] / 99.0, base[t] / y[t], rtol=1e-12)


def test_prepare_dataset_shapes_burn_in_and_alignment():
    prepared = prepare_dataset()

    assert prepared.x.shape[1] == 15
    assert prepared.x.shape[0] == prepared.y.shape[0] == prepared.dates.shape[0]
    assert np.isfinite(prepared.x).all()
    assert np.isfinite(prepared.y).all()

    # Sample starts 1927-01 (first complete predictor row) + 36-month
    # burn-in -> 1930-01, matching the MATLAB Y(37:end) drop.
    assert prepared.dates[0] == 193001
    assert prepared.dates[-1] >= 202412


def test_prepare_dataset_matches_gydata_through_same_pipeline():
    """Wire the authors' X, Y through our standardization and compare.

    Phase-1 tests pinned construction parity on raw values; this test pins
    the *plumbing* (ordering of lag, standardization, burn-in) by running
    the GYdata fixture through the same volstd/target-scale code.
    """
    if not FIXTURE.exists():
        pytest.skip("gydata_slice.npz fixture not present")
    data = np.load(FIXTURE)
    x_gy, y_gy, dates_gy = data["X"], data["Y"], data["dates"]

    # MATLAB order: append lag-mkt, volstd X, vol-scale Y, drop 36 rows.
    x_full = np.column_stack([x_gy, np.concatenate([[np.nan], y_gy[:-1]])])
    x_std = volstd_expanding(x_full, min_obs=BURN_IN_MONTHS)[BURN_IN_MONTHS:]
    y_std = target_vol_scale(y_gy, months=12)[BURN_IN_MONTHS:]
    dates_exp = dates_gy[BURN_IN_MONTHS:]

    prepared = prepare_dataset()
    overlap = np.isin(prepared.dates, dates_exp)
    sub_dates = prepared.dates[overlap]
    np.testing.assert_array_equal(sub_dates, dates_exp)

    # Tolerances: raw-data vintage rounding (~1e-4) gets amplified by
    # division through small expanding stds (e.g. dfr std ~0.013 -> 6e-3).
    # A plumbing error (wrong lag/order) would show as O(1) differences on
    # unit-scale standardized data, far above 1e-2. bm is exact pre-2009
    # and carries the documented vintage restatement after (0.15 raw /
    # ~0.27 expanding std -> ~0.56 standardized).
    bm_col = GW_PREDICTOR_ORDER.index("bm")
    # ntis also carries minor vintage revisions (raw ~7e-4 / std ~0.025).
    ntis_col = GW_PREDICTOR_ORDER.index("ntis")
    # lag_mkt: GYdata's first entry is NaN (lagmatrix artifact) while our
    # longer sample has the real 1926-12 return, so every expanding std
    # differs by one observation. The effect peaks early (~3e-2 in 1930)
    # and decays as the window grows — assert exactly that signature.
    lag_mkt_col = 14
    ours_x = prepared.x[overlap]
    pre_revision = sub_dates < 200904
    for j in range(15):
        diff = np.abs(ours_x[:, j] - x_std[:, j])
        if j == bm_col:
            assert diff[pre_revision].max() < 1e-6, (
                f"bm pre-revision: {diff[pre_revision].max():.2e}"
            )
            assert diff[~pre_revision].max() < 0.7, (
                f"bm post-revision: {diff[~pre_revision].max():.2e}"
            )
        elif j == lag_mkt_col:
            assert diff.max() < 5e-2, f"lag_mkt: {diff.max():.2e}"
            assert diff[sub_dates >= 194001].max() < 1e-2, "lag_mkt diff must decay"
        else:
            tol = 5e-2 if j == ntis_col else 1e-2
            assert diff.max() < tol, f"column {j}: standardized max|diff|={diff.max():.2e}"

    np.testing.assert_allclose(prepared.y[overlap], y_std, atol=5e-3)
