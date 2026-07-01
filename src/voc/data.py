"""Raw data loading.

The two CSVs use different decimal conventions: the Goyal-Welch predictor
file uses European decimal commas, the Fama-French file uses points. Loading
them with the wrong setting yields object-dtype columns that silently turn
into NaN downstream, so each loader hard-codes its convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from voc.constants import DATE_COL, GW_PREDICTOR_ORDER, PREDICTORS_FILE, TARGET_FILE


def project_root() -> Path:
    if "google.colab" in sys.modules:
        return Path.cwd()
    return Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    candidates = [project_root() / "data", project_root() / "src" / "data"]
    for candidate in candidates:
        if (candidate / PREDICTORS_FILE).exists() and (candidate / TARGET_FILE).exists():
            return candidate
    raise FileNotFoundError(
        f"Data files not found. Expected both {PREDICTORS_FILE} and {TARGET_FILE} "
        "under one of: " + ", ".join(str(c) for c in candidates)
    )


def _load_sorted(path: Path, *, decimal: str, thousands: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, decimal=decimal, thousands=thousands)
    return frame.sort_values(DATE_COL, ignore_index=True)


def load_raw_predictors() -> pd.DataFrame:
    """Raw Goyal-Welch monthly data (1871-2024), European-encoded.

    Decimal comma AND dot thousands separator: the S&P level appears as
    "5.881,63" once it crosses 1000, so decimal="," alone leaves Index as
    strings.
    """
    return _load_sorted(resolve_data_dir() / PREDICTORS_FILE, decimal=",", thousands=".")


def load_ff_returns() -> pd.DataFrame:
    """Fama-French monthly factors in percent (1926-07 onward)."""
    return _load_sorted(resolve_data_dir() / TARGET_FILE, decimal=".")


def build_gw_predictors(raw: pd.DataFrame) -> pd.DataFrame:
    """Construct the 14 Goyal-Welch predictors from raw columns, UNlagged.

    Each value sits at its observation date; `assemble_xy` applies the
    one-month availability lag. Valuation ratios are logs: price and
    dividend levels span orders of magnitude over 150 years, and the log
    ratio is the near-stationary object the literature uses. `dy` divides
    by the PRIOR month's price (yield convention: dividends received over
    the price you could have paid).

    Verified column-by-column against GYdata.mat on 192701-202012; only
    `bm` after 2009-03 differs (Goyal-Welch restated book values between
    vintages).
    """
    raw = raw.set_index(DATE_COL)
    log_d12 = np.log(raw["D12"])
    log_e12 = np.log(raw["E12"])
    log_index = np.log(raw["Index"])

    predictors = pd.DataFrame(
        {
            "dfy": raw["BAA"] - raw["AAA"],
            "infl": raw["infl"],
            "svar": raw["svar"],
            "de": log_d12 - log_e12,
            "lty": raw["lty"],
            "tms": raw["lty"] - raw["tbl"],
            "tbl": raw["tbl"],
            "dfr": raw["corpr"] - raw["ltr"],
            "dp": log_d12 - log_index,
            "dy": log_d12 - log_index.shift(1),
            "ltr": raw["ltr"],
            "ep": log_e12 - log_index,
            "bm": raw["b/m"],
            "ntis": raw["ntis"],
        }
    )
    return predictors[GW_PREDICTOR_ORDER]


def build_market_excess_return(raw: pd.DataFrame) -> pd.Series:
    """Monthly market excess return: CRSP value-weighted return minus T-bill.

    This is GYdata's Y (established empirically: max |diff| 5e-5 against the
    fixture, versus 1.4e-2 for raw CRSP and 5.5e-2 for FF Mkt-RF).
    """
    raw = raw.set_index(DATE_COL)
    return (raw["CRSP_SPvw"] - raw["Rfree"]).rename("mkt_excess")


def assemble_xy() -> tuple[pd.DataFrame, pd.Series]:
    """Model inputs: X[N, 15] = lagged GW predictors + lagged return, Y[N].

    Row t of X contains only information observable at t-1, matching
    GYdata.mat where X ships pre-lagged. Rows with any missing input are
    dropped, so the usable sample starts when all 14 predictors exist
    (1927-01 with the current vintage).
    """
    raw = load_raw_predictors()
    y = build_market_excess_return(raw)

    x = build_gw_predictors(raw).shift(1)
    x["lag_mkt"] = y.shift(1)

    valid = x.notna().all(axis=1) & y.notna()
    return x.loc[valid], y.loc[valid]
