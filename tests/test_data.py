"""Tests for raw data loading.

The predictor CSV uses European decimal commas ("0,26"). If parsed with the
wrong decimal setting, pandas silently produces object-dtype columns and the
whole pipeline degrades to NaNs. These tests pin float64 dtypes and known
values so a data refresh cannot regress parsing unnoticed.
"""

import numpy as np

from voc.data import load_ff_returns, load_raw_predictors

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


def test_ff_returns_parse():
    ff = load_ff_returns()

    assert {"yyyymm", "Mkt-RF", "RF"} <= set(ff.columns)
    assert ff["Mkt-RF"].dtype == np.float64
    assert ff["yyyymm"].iloc[0] == 192607
    assert ff["yyyymm"].is_monotonic_increasing
    # Fama-French factors are in percent: monthly |Mkt-RF| stays well under 50.
    assert ff["Mkt-RF"].abs().max() < 50
