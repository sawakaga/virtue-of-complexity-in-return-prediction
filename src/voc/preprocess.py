"""Standardization pipeline: ports of volstdbwd.m and the target vol scaling.

Two different scalings, both strictly causal, both scale-only (demean=0 in
the MATLAB driver — the sign of a predictor carries economic meaning, and an
estimated mean would inject one more forward-looking statistic):

- Predictors: divide by an EXPANDING std (all history up to and including
  the current row). Slow-moving; only the scale is normalized so that the
  RFF projection gamma*W@x operates on comparable magnitudes per column.
- Target: divide by the root mean square of the previous 12 monthly
  returns, excluding the current one. Fast-moving volatility proxy; using
  lagged values only keeps the realized return at t out of its own scaling
  (no look-ahead), so the scaled value is a tradable quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voc.constants import BURN_IN_MONTHS, TARGET_VOL_MONTHS
from voc.data import assemble_xy


@dataclass(slots=True)
class PreparedData:
    """Model-ready sample: x[t] is observable at t-1, y[t] realized at t."""

    x: np.ndarray  # [N, 15] float64, vol-standardized
    y: np.ndarray  # [N] float64, vol-scaled excess return
    dates: np.ndarray  # [N] int64 yyyymm


def volstd_expanding(values: np.ndarray, *, min_obs: int = BURN_IN_MONTHS) -> np.ndarray:
    """Divide each row by the expanding NaN-aware std (ddof=1), volstdbwd.m.

    Rows 0..min_obs-1 are divided by the std of the first min_obs rows;
    row t >= min_obs by the std of rows 0..t INCLUSIVE (MATLAB X(1:t,:)).
    No demeaning of the values themselves (the std is mean-centered
    internally, but the data keeps its level).
    """
    values = np.asarray(values, dtype=np.float64)
    n_rows = values.shape[0]
    if n_rows < min_obs:
        raise ValueError(f"need at least {min_obs} rows, got {n_rows}")

    # Expanding NaN-aware variance via prefix sums in float64. The
    # E[x^2] - E[x]^2 form risks catastrophic cancellation in low
    # precision; float64 with economically-scaled inputs is safe, and the
    # ddof=1 tests pin the result against np.std on clean windows.
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    count = np.cumsum(finite, axis=0).astype(np.float64)
    s1 = np.cumsum(filled, axis=0)
    s2 = np.cumsum(filled * filled, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / count) / (count - 1.0)
        std = np.sqrt(np.maximum(var, 0.0))

    # Rows below min_obs all use the std of the initial block.
    std[:min_obs] = std[min_obs - 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return values / std


def target_vol_scale(y: np.ndarray, *, months: int = TARGET_VOL_MONTHS) -> np.ndarray:
    """Scale returns by trailing realized vol: y[t] / sqrt(mean(y[t-12..t-1]^2)).

    Strictly lagged second moment — no mean subtraction (it is a raw RMS,
    not a variance) and no current-month information. The first `months`
    entries are NaN.
    """
    y = np.asarray(y, dtype=np.float64)
    sq = y * y
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    out = np.full_like(y, np.nan)
    # mean of sq[t-months .. t-1] = (csum[t] - csum[t-months]) / months
    trailing_mean = (csum[months:-1] - csum[: -months - 1]) / months
    out[months:] = y[months:] / np.sqrt(trailing_mean)
    return out


def prepare_dataset() -> PreparedData:
    """Full MATLAB preprocessing: lag, standardize, burn-in drop.

    Order matters and mirrors tryrff_v2_function_for_each_sim.m: lag_mkt is
    appended BEFORE volstd (so the lagged return is also expanding-vol
    standardized), the target uses its own 12-month scaling, and the first
    36 rows are dropped because the expanding std is too noisy there.
    """
    x_frame, y_series = assemble_xy()

    x_std = volstd_expanding(x_frame.to_numpy(dtype=np.float64))
    y_std = target_vol_scale(y_series.to_numpy(dtype=np.float64))
    dates = x_frame.index.to_numpy(dtype=np.int64)

    keep = slice(BURN_IN_MONTHS, None)
    return PreparedData(x=x_std[keep], y=y_std[keep], dates=dates[keep])
