"""Tests for raw data loading.

The predictor CSV uses European decimal commas ("0,26"). If parsed with the
wrong decimal setting, pandas silently produces object-dtype columns and the
whole pipeline degrades to NaNs. These tests pin float64 dtypes and known
values so a data refresh cannot regress parsing unnoticed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from voc.constants import GW_PREDICTOR_ORDER
from voc.data import (
    assemble_xy,
    build_gw_predictors,
    build_market_excess_return,
    load_ff_returns,
    load_raw_predictors,
)

FIXTURE = Path(__file__).parent / "fixtures" / "gydata_slice.npz"

RAW_PREDICTOR_COLUMNS = [
    "yyyymm",
    "Index",
    "D12",
    "E12",
    "b/m",
    "tbl",
    "AAA",
    "BAA",
    "lty",
    "ntis",
    "Rfree",
    "infl",
    "ltr",
    "corpr",
    "svar",
    "csp",
    "CRSP_SPvw",
    "CRSP_SPvwx",
]


def test_raw_predictors_parse_decimal_commas_to_float64():
    raw = load_raw_predictors()

    assert list(raw.columns) == RAW_PREDICTOR_COLUMNS

    numeric = raw.drop(columns=["yyyymm"])
    object_cols = [col for col, dtype in numeric.dtypes.items() if dtype != np.float64]
    assert not object_cols, f"non-float columns (decimal parsing broke): {object_cols}"

    # Known value: S&P index level for Jan 1871 is 4.44 — if decimal commas
    # were mis-parsed this would be NaN or 444.
    jan_1871_index = raw.loc[raw["yyyymm"] == 187101, "Index"].iloc[0]
    assert abs(jan_1871_index - 4.44) < 1e-9

    # Known value with a thousands separator: Dec 2024 is "5.881,63".
    # Guards both failure modes: missing thousands="." (string/NaN) and a
    # thousands separator swallowing a decimal point (588163.0).
    dec_2024_index = raw.loc[raw["yyyymm"] == 202412, "Index"].iloc[0]
    assert abs(dec_2024_index - 5881.63) < 1e-9


def test_raw_predictors_date_coverage():
    raw = load_raw_predictors()

    assert raw["yyyymm"].dtype == np.int64
    assert raw["yyyymm"].is_monotonic_increasing
    assert raw["yyyymm"].iloc[0] == 187101
    assert raw["yyyymm"].iloc[-1] >= 202412


@pytest.fixture(scope="module")
def gydata():
    """X, Y, dates exported from the authors' GYdata.mat (192701-202012).

    X holds the 14 constructed Goyal-Welch predictors, already lagged one
    month relative to Y. Y is the CRSP value-weighted excess return
    (CRSP_SPvw - Rfree, established empirically: max diff 5e-5 vs 1.4e-2
    for the next-best candidate).
    """
    if not FIXTURE.exists():
        pytest.skip("gydata_slice.npz fixture not present")
    data = np.load(FIXTURE)
    return data["X"], data["Y"], data["dates"]


# Per-column absolute tolerances against GYdata (~2021 vintage) using the
# 2024-vintage CSV. Tight bounds = formula verification; looser bounds
# reflect source-data rounding (CSV stores fewer decimals than the .mat).
COLUMN_ATOL = {
    "dfy": 1e-12,
    "infl": 1e-6,
    "svar": 1e-5,
    "de": 2e-4,
    "lty": 1e-4,
    "tms": 1e-4,
    "tbl": 1e-12,
    "dfr": 2e-4,
    "dp": 2e-4,
    "dy": 2e-4,
    "ltr": 1e-4,
    "ep": 2e-4,
    "bm": 1e-9,  # matches to CSV rounding until 2009-03; revised rows tested separately
    "ntis": 1e-3,
}
# Goyal-Welch restated book-to-market for 2009-04..2020-12 between vintages.
BM_REVISED_FROM = 200904
BM_REVISED_ATOL = 0.15


def test_gw_predictor_construction_matches_gydata(gydata):
    x_gy, _, dates = gydata
    expected = pd.DataFrame(x_gy, index=dates, columns=GW_PREDICTOR_ORDER)

    raw = load_raw_predictors()
    constructed = build_gw_predictors(raw).shift(1).loc[dates]

    for i, col in enumerate(GW_PREDICTOR_ORDER):
        diff = (constructed[col] - expected[col]).abs()
        if col == "bm":
            pre = diff[diff.index < BM_REVISED_FROM]
            post = diff[diff.index >= BM_REVISED_FROM]
            assert pre.max() <= COLUMN_ATOL[col], f"bm pre-revision mismatch: {pre.max():.2e}"
            assert post.max() <= BM_REVISED_ATOL, f"bm revision drift too large: {post.max():.2e}"
        else:
            assert diff.max() <= COLUMN_ATOL[col], f"{col} (#{i}): max|diff|={diff.max():.2e}"


def test_market_excess_return_matches_gydata_y(gydata):
    _, y_gy, dates = gydata
    y = build_market_excess_return(load_raw_predictors()).loc[dates]
    assert (y - pd.Series(y_gy, index=dates)).abs().max() < 1e-4


def test_assemble_xy_appends_lagged_return_and_aligns(gydata):
    x_gy, y_gy, dates = gydata
    x, y = assemble_xy()

    assert list(x.columns) == [*GW_PREDICTOR_ORDER, "lag_mkt"]
    assert x.shape[1] == 15
    assert x.index.equals(y.index)
    assert not x.isna().to_numpy().any()
    assert not y.isna().to_numpy().any()

    # lag_mkt at t must equal the realized excess return at t-1 — verify on
    # the GYdata overlap where both series exist.
    overlap = x.index.intersection(pd.Index(dates))
    y_map = pd.Series(y_gy, index=dates)
    lag_expected = y_map.shift(1).loc[overlap].dropna()
    diff = (x.loc[lag_expected.index, "lag_mkt"] - lag_expected).abs()
    assert diff.max() < 1e-4

    # Coverage: usable sample should reach back at least to GYdata's start
    # and extend to 2024 with the newer vintage.
    assert x.index[0] <= 192701
    assert x.index[-1] >= 202412


def test_ff_returns_parse():
    ff = load_ff_returns()

    assert {"yyyymm", "Mkt-RF", "RF"} <= set(ff.columns)
    assert ff["Mkt-RF"].dtype == np.float64
    assert ff["yyyymm"].iloc[0] == 192607
    assert ff["yyyymm"].is_monotonic_increasing
    # Fama-French factors are in percent: monthly |Mkt-RF| stays well under 50.
    assert ff["Mkt-RF"].abs().max() < 50
