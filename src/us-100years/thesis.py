from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge
from tqdm import tqdm

PREDICTORS_FILE = "15-predictors.csv"
TARGET_FILE = "fama-french-return.csv"
MIN_PERIODS = 36
ROLLING_WINDOW = 12
DATE_COL = "yyyymm"
TARGET_COL = "Mkt-RF"
INDEX_COL = "Index"
START_DATE = 193001
MAX_RFF_FEATURES = 12000
TRAINING_WINDOWS = [12, 60, 120]
RIDGE_ALPHAS = [10 ** p for p in range(-3, 4)]
GAMMA = 2.0
RANDOM_SEED = 123


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    candidates = [
        project_root() / "data",
        project_root() / "src" / "data",
    ]
    for candidate in candidates:
        if (candidate / PREDICTORS_FILE).exists() and (candidate / TARGET_FILE).exists():
            return candidate
    raise FileNotFoundError(
        "Data files not found. Expected both "
        f"{PREDICTORS_FILE} and {TARGET_FILE} under one of: "
        + ", ".join(str(c) for c in candidates)
    )


def load_csv(path: Path, *, decimal: str) -> pd.DataFrame:
    return pd.read_csv(path, decimal=decimal)


def require_columns(df: pd.DataFrame, columns: list[str], *, name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def standardize_expanding(
    df: pd.DataFrame, columns: list[str], *, min_periods: int
) -> pd.DataFrame:
    standardized = df.copy()
    for col in columns:
        expanding_mean = standardized[col].expanding(min_periods=min_periods).mean()
        expanding_std = standardized[col].expanding(min_periods=min_periods).std()
        standardized[col] = (standardized[col] - expanding_mean) / expanding_std
    return standardized


def standardize_rolling(
    df: pd.DataFrame, column: str, *, window: int
) -> pd.DataFrame:
    standardized = df.copy()
    rolling_mean = standardized[column].rolling(window=window).mean()
    rolling_std = standardized[column].rolling(window=window).std()
    standardized[column] = (standardized[column] - rolling_mean) / rolling_std
    return standardized


def rff_feature_counts(window: int, *, max_features: int) -> list[int]:
    if window <= 0:
        raise ValueError("window must be positive")
    return [k * window for k in range(1, max_features // window + 1)]


def build_lagged_matrix(
    data: pd.DataFrame, predictor_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    predictors = data[predictor_cols]
    target = data[TARGET_COL]

    # X_t = S_{t-1}, y_t = R_t
    x_lagged = predictors.shift(1)
    y_aligned = target

    valid = x_lagged.notna().all(axis=1) & y_aligned.notna()
    x_lagged = x_lagged.loc[valid]
    y_aligned = y_aligned.loc[valid]

    return x_lagged, y_aligned


def run_rff_ridge_oos(
    data: pd.DataFrame,
    predictor_cols: list[str],
    *,
    windows: list[int],
    max_features: int,
    gamma: float,
    alphas: list[float],
    seed: int,
) -> pd.DataFrame:
    x_lagged, y_aligned = build_lagged_matrix(data, predictor_cols)
    dates = y_aligned.index

    results = []
    features_to_windows: dict[int, list[int]] = {}
    for window in windows:
        feature_counts = rff_feature_counts(window, max_features=max_features)
        for n_features in feature_counts:
            features_to_windows.setdefault(n_features, []).append(window)

    def run_feature_config(n_features: int) -> list[dict]:
        rff = RBFSampler(
            gamma=gamma, n_components=n_features, random_state=seed
        )
        z_all = rff.fit_transform(x_lagged.to_numpy())
        local_results = []

        for window in features_to_windows[n_features]:
            for alpha in alphas:
                for t in range(window, len(y_aligned) - 1):
                    x_train = z_all[t - window : t]
                    y_train = y_aligned.iloc[t - window + 1 : t + 1].to_numpy()

                    model = Ridge(alpha=alpha)
                    model.fit(x_train, y_train)

                    x_forecast = z_all[t : t + 1]
                    y_pred = float(model.predict(x_forecast)[0])
                    y_true = float(y_aligned.iloc[t + 1])

                    local_results.append(
                        {
                            "date": int(dates[t + 1]),
                            "window": window,
                            "gamma": gamma,
                            "n_features": n_features,
                            "alpha": alpha,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "strategy_return": y_pred * y_true,
                        }
                    )

        return local_results

    feature_configs = list(features_to_windows.keys())
    config_bar = tqdm(feature_configs, desc="RFF features", unit="config")
    n_jobs = max(1, (os.cpu_count() or 2) - 1)
    parallel_results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(run_feature_config)(n_features)
        for n_features in config_bar
    )

    for chunk in parallel_results:
        results.extend(chunk)

    return pd.DataFrame(results)


def main() -> None:
    data_dir = resolve_data_dir()

    predictors = load_csv(data_dir / PREDICTORS_FILE, decimal=",")
    """ Fama French Factors decimal is ."""
    target = load_csv(data_dir / TARGET_FILE, decimal=".")

    require_columns(predictors, [DATE_COL], name="predictors")
    require_columns(target, [DATE_COL, TARGET_COL], name="target")

    data = predictors.merge(target, on=DATE_COL, how="inner")
    data = data.set_index(DATE_COL).sort_index()

    predictor_cols = [
        col
        for col in predictors.columns
        if col not in {DATE_COL, INDEX_COL, "csp"}
    ]
    # Drop csp: it contains many missing values across the sample and
    # truncates the usable window when requiring complete predictor data.

    # Keep only predictors + target to avoid unintended use of extra columns.
    keep_cols = predictor_cols + [TARGET_COL]
    data = data[keep_cols]

    data = standardize_expanding(data, predictor_cols, min_periods=MIN_PERIODS)
    data = standardize_rolling(data, TARGET_COL, window=ROLLING_WINDOW)

    required_cols = predictor_cols + [TARGET_COL]
    data = data.dropna(subset=required_cols)
    data = data.loc[data.index >= START_DATE]

    print("Input data types")
    print(predictors.dtypes)
    print(target.dtypes)
    print()

    print("Data range after standardization")
    print(f"Start date: {data.index.min()}")
    print(f"End date:   {data.index.max()}")
    print(f"Final sample size: {len(data)} months")
    print(f"Expected start date: {START_DATE}")
    print(f"Actual start date:   {data.index.min()}")
    print()
    print(data.describe())

    results = run_rff_ridge_oos(
        data,
        predictor_cols,
        windows=TRAINING_WINDOWS,
        max_features=MAX_RFF_FEATURES,
        gamma=GAMMA,
        alphas=RIDGE_ALPHAS,
        seed=RANDOM_SEED,
    )

    output_path = project_root() / "artifacts" / "oos_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print()
    print(f"Wrote out-of-sample predictions to {output_path}")


if __name__ == "__main__":
    main()
