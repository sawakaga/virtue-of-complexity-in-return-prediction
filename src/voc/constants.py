"""Paper constants from the MATLAB reference implementation.

Single source of truth: tests and modules import these instead of re-typing
literals, so a constant can only be wrong in one place.

Sources (MATLAB reference, Kelly-Malamud-Zhou 2023):
- predictions_main.m: gamma, training windows, number of seeds
- tryrff_v2_function_for_each_sim.m: maxP, lambda grid, burn-in
- rffexhibits_function.m: subsamples, percentile list
"""

# RFF scale parameter: Z = [cos(gamma * W @ x); sin(gamma * W @ x)], W ~ N(0, I).
GAMMA = 2.0

# Maximum total RFF feature count (cos and sin blocks together).
MAX_P = 12_000

# Ridge shrinkage grid; effective penalty in the objective is lam * T.
LAMBDA_GRID = [10.0**p for p in range(-3, 4)]

# Rolling training window lengths in months.
TRAINING_WINDOWS = [12, 60, 120]

# Observations consumed by the expanding volatility standardization before
# the sample is usable (MATLAB drops rows 1..36).
BURN_IN_MONTHS = 36

# Trailing window (months) for target volatility scaling.
TARGET_VOL_MONTHS = 12

RANDOM_SEED = 123

# Evaluation subsamples (inclusive year bounds) from rffexhibits_function.m.
SUBSAMPLES = [(1926, 2020), (1926, 1974), (1975, 2020)]

# Cross-seed percentile bands reported by the paper.
PERCENTILES = [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]

# Constructed Goyal-Welch predictors in GYdata.mat column order
# (tryrff_v2_function_for_each_sim_DropOnePredictor.m, X_columns).
# The model input is these 14, lagged one month, plus lag_mkt.
GW_PREDICTOR_ORDER = [
    "dfy",  # default yield spread: BAA - AAA
    "infl",  # CPI inflation
    "svar",  # stock variance (sum of squared daily returns)
    "de",  # log dividend-earnings ratio: log(D12) - log(E12)
    "lty",  # long-term government bond yield
    "tms",  # term spread: lty - tbl
    "tbl",  # 3-month T-bill rate
    "dfr",  # default return spread: corpr - ltr
    "dp",  # log dividend-price ratio: log(D12) - log(Index)
    "dy",  # log dividend yield: log(D12) - log(lagged Index)
    "ltr",  # long-term government bond return
    "ep",  # log earnings-price ratio: log(E12) - log(Index)
    "bm",  # book-to-market ratio
    "ntis",  # net equity expansion
]

PREDICTORS_FILE = "15-predictors.csv"
TARGET_FILE = "fama-french-return.csv"
DATE_COL = "yyyymm"
