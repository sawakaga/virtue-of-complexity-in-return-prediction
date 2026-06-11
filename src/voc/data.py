"""Raw data loading.

The two CSVs use different decimal conventions: the Goyal-Welch predictor
file uses European decimal commas, the Fama-French file uses points. Loading
them with the wrong setting yields object-dtype columns that silently turn
into NaN downstream, so each loader hard-codes its convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from voc.constants import DATE_COL, PREDICTORS_FILE, TARGET_FILE


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
