from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
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
RIDGE_ALPHAS = [10**p for p in range(-3, 4)]
GAMMA = 2.0
RANDOM_SEED = 123

SolverPolicy = Literal["auto", "primal", "dual"]


@dataclass(slots=True)
class GPUFitConfig:
    windows: list[int]
    max_features: int
    gamma: float
    alphas: list[float]
    seed: int
    dtype: torch.dtype
    device: torch.device
    solver_policy: SolverPolicy = "auto"
    fit_only: bool = True
    chunk_size_windows: int = 128
    chunk_size_features: int | None = None
    profile: bool = False
    deterministic: bool = False


@dataclass(slots=True)
class FeatureProfile:
    n_features: int
    rff_time_s: float
    window_build_time_s: float
    solve_time_s: float
    total_time_s: float
    windows_processed: int
    chunks_processed: int
    solve_calls: int
    primal_solve_calls: int
    dual_solve_calls: int
    cuda_memory_allocated_bytes: int | None
    cuda_max_memory_allocated_bytes: int | None


@dataclass(slots=True)
class FitCoreMetrics:
    device: str
    dtype: str
    n_samples: int
    n_predictors: int
    total_features: int
    total_windows_processed: int
    total_chunks_processed: int
    total_solve_calls: int
    total_primal_solve_calls: int
    total_dual_solve_calls: int
    total_time_s: float
    checksum: float
    feature_profiles: list[FeatureProfile]


def project_root() -> Path:
    if "google.colab" in str(get_ipython()):
        return Path(Path.cwd())
    return Path(__file__).resolve().parents[2]


def resolve_data_dir() -> Path:
    candidates = [
        project_root() / "data",
        project_root() / "src" / "data",
    ]
    for candidate in candidates:
        if (candidate / PREDICTORS_FILE).exists() and (
            candidate / TARGET_FILE
        ).exists():
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


def standardize_rolling(df: pd.DataFrame, column: str, *, window: int) -> pd.DataFrame:
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


def prepare_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = resolve_data_dir()

    predictors = load_csv(data_dir / PREDICTORS_FILE, decimal=",")
    target = load_csv(data_dir / TARGET_FILE, decimal=".")

    require_columns(predictors, [DATE_COL], name="predictors")
    require_columns(target, [DATE_COL, TARGET_COL], name="target")

    data = predictors.merge(target, on=DATE_COL, how="inner")
    data = data.set_index(DATE_COL).sort_index()

    predictor_cols = [
        col for col in predictors.columns if col not in {DATE_COL, INDEX_COL, "csp"}
    ]

    keep_cols = predictor_cols + [TARGET_COL]
    data = data[keep_cols]

    data = standardize_expanding(data, predictor_cols, min_periods=MIN_PERIODS)
    data = standardize_rolling(data, TARGET_COL, window=ROLLING_WINDOW)

    required_cols = predictor_cols + [TARGET_COL]
    data = data.dropna(subset=required_cols)
    data = data.loc[data.index >= START_DATE]

    x_lagged, y_aligned = build_lagged_matrix(data, predictor_cols)

    return (
        x_lagged.to_numpy(dtype=np.float32),
        y_aligned.to_numpy(dtype=np.float32),
        y_aligned.index.to_numpy(dtype=np.int64),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this host.")
        return torch.device("cuda")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")


def generate_rff_weights(
    n_features: int,
    input_dim: int,
    *,
    gamma: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w = torch.normal(
        mean=0.0,
        std=(2 * gamma) ** 0.5,
        size=(input_dim, n_features),
        generator=generator,
    ).to(device=device, dtype=dtype)
    b = torch.rand(n_features, generator=generator).to(device=device, dtype=dtype) * (
        2 * np.pi
    )
    return w, b


def rff_transform(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    projection = x @ w + b
    z = torch.cos(projection)
    z *= (2.0 / w.shape[1]) ** 0.5
    return z


def build_window_tensors(
    z_all: torch.Tensor,
    y_all: torch.Tensor,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if window < 1:
        raise ValueError("window must be >= 1")
    if y_all.shape[0] <= window:
        raise ValueError("window must be smaller than sample length")

    # z_windows shape: [K, T, F], y_windows shape: [K, T], where K = N - T.
    z_windows = z_all.unfold(0, window, 1)[:-1].transpose(1, 2)
    y_windows = y_all.unfold(0, window, 1)[:-1]
    return z_windows, y_windows


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _cuda_memory_snapshot(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.memory_allocated(device=device)),
        int(torch.cuda.max_memory_allocated(device=device)),
    )


def _select_solver(
    policy: SolverPolicy, n_features: int, window: int
) -> Literal["primal", "dual"]:
    if policy == "primal":
        return "primal"
    if policy == "dual":
        return "dual"
    return "primal" if n_features <= window else "dual"


def _solve_primal(
    z_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
    alphas: Iterable[float],
) -> dict[float, torch.Tensor]:
    zt = z_chunk.transpose(1, 2)
    xtx = zt @ z_chunk
    xty = zt @ y_chunk.unsqueeze(-1)

    batch, features, _ = xtx.shape
    eye = torch.eye(features, device=xtx.device, dtype=xtx.dtype).expand(batch, -1, -1)

    betas = {}
    for alpha in alphas:
        system = xtx + float(alpha) * eye
        chol, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if torch.any(info != 0):
            raise RuntimeError("Primal Cholesky failed for one or more batches.")
        betas[float(alpha)] = torch.cholesky_solve(xty, chol).squeeze(-1)
    return betas


def _dual_beta_from_a(
    z_chunk: torch.Tensor,
    a: torch.Tensor,
    *,
    chunk_size_features: int | None,
) -> torch.Tensor:
    zt = z_chunk.transpose(1, 2)
    if chunk_size_features is None or chunk_size_features >= zt.shape[1]:
        return (zt @ a).squeeze(-1)

    parts: list[torch.Tensor] = []
    for start in range(0, zt.shape[1], chunk_size_features):
        end = min(start + chunk_size_features, zt.shape[1])
        parts.append((zt[:, start:end, :] @ a).squeeze(-1))
    return torch.cat(parts, dim=1)


def _solve_dual(
    z_chunk: torch.Tensor,
    y_chunk: torch.Tensor,
    alphas: Iterable[float],
    *,
    chunk_size_features: int | None,
) -> dict[float, torch.Tensor]:
    k = z_chunk @ z_chunk.transpose(1, 2)
    ycol = y_chunk.unsqueeze(-1)

    batch, t_dim, _ = k.shape
    eye = torch.eye(t_dim, device=k.device, dtype=k.dtype).expand(batch, -1, -1)

    betas = {}
    for alpha in alphas:
        system = k + float(alpha) * eye
        chol, info = torch.linalg.cholesky_ex(system, check_errors=False)
        if torch.any(info != 0):
            raise RuntimeError("Dual Cholesky failed for one or more batches.")
        a = torch.cholesky_solve(ycol, chol)
        betas[float(alpha)] = _dual_beta_from_a(
            z_chunk,
            a,
            chunk_size_features=chunk_size_features,
        )
    return betas


def run_fit_core_gpu(
    x_lagged: np.ndarray,
    y_aligned: np.ndarray,
    config: GPUFitConfig,
) -> FitCoreMetrics:
    if config.chunk_size_windows < 1:
        raise ValueError("chunk_size_windows must be >= 1")
    if config.chunk_size_features is not None and config.chunk_size_features < 1:
        raise ValueError("chunk_size_features must be >= 1 when provided")

    if config.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=config.device)

    deterministic_prev = torch.are_deterministic_algorithms_enabled()
    if config.deterministic:
        torch.use_deterministic_algorithms(True)
        if config.device.type == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    feature_counts = rff_feature_counts(
        min(config.windows), max_features=config.max_features
    )

    x_t = torch.tensor(x_lagged, device=config.device, dtype=config.dtype)
    y_t = torch.tensor(y_aligned, device=config.device, dtype=config.dtype)

    feature_profiles: list[FeatureProfile] = []
    checksum = 0.0
    total_windows_processed = 0
    total_chunks_processed = 0
    total_solve_calls = 0
    total_primal_solve_calls = 0
    total_dual_solve_calls = 0

    run_start = time.perf_counter()

    try:
        for n_features in tqdm(feature_counts, desc="RFF features", unit="config"):
            feature_start = time.perf_counter()

            rff_start = time.perf_counter()
            w_t, b_t = generate_rff_weights(
                n_features,
                x_t.shape[1],
                gamma=config.gamma,
                seed=config.seed,
                device=config.device,
                dtype=config.dtype,
            )
            z_all = rff_transform(x_t, w_t, b_t)
            _synchronize(config.device)
            rff_time = time.perf_counter() - rff_start

            window_build_time = 0.0
            solve_time = 0.0
            windows_processed = 0
            chunks_processed = 0
            solve_calls = 0
            primal_solve_calls = 0
            dual_solve_calls = 0

            for window in config.windows:
                if y_t.shape[0] <= window:
                    continue

                build_start = time.perf_counter()
                z_windows, y_windows = build_window_tensors(z_all, y_t, window)
                _synchronize(config.device)
                window_build_time += time.perf_counter() - build_start
                windows_processed += 1

                solver = _select_solver(config.solver_policy, n_features, window)
                total_k = z_windows.shape[0]
                for start in range(0, total_k, config.chunk_size_windows):
                    end = min(start + config.chunk_size_windows, total_k)
                    z_chunk = z_windows[start:end]
                    y_chunk = y_windows[start:end]

                    solve_start = time.perf_counter()
                    if solver == "primal":
                        betas = _solve_primal(z_chunk, y_chunk, config.alphas)
                        primal_solve_calls += (end - start) * len(config.alphas)
                    else:
                        betas = _solve_dual(
                            z_chunk,
                            y_chunk,
                            config.alphas,
                            chunk_size_features=config.chunk_size_features,
                        )
                        dual_solve_calls += (end - start) * len(config.alphas)
                    _synchronize(config.device)
                    solve_time += time.perf_counter() - solve_start

                    for beta in betas.values():
                        checksum += float(beta.sum().item())

                    chunks_processed += 1
                    solve_calls += (end - start) * len(config.alphas)

            current_mem, max_mem = _cuda_memory_snapshot(config.device)
            feature_total_time = time.perf_counter() - feature_start

            feature_profiles.append(
                FeatureProfile(
                    n_features=n_features,
                    rff_time_s=rff_time,
                    window_build_time_s=window_build_time,
                    solve_time_s=solve_time,
                    total_time_s=feature_total_time,
                    windows_processed=windows_processed,
                    chunks_processed=chunks_processed,
                    solve_calls=solve_calls,
                    primal_solve_calls=primal_solve_calls,
                    dual_solve_calls=dual_solve_calls,
                    cuda_memory_allocated_bytes=current_mem,
                    cuda_max_memory_allocated_bytes=max_mem,
                )
            )

            total_windows_processed += windows_processed
            total_chunks_processed += chunks_processed
            total_solve_calls += solve_calls
            total_primal_solve_calls += primal_solve_calls
            total_dual_solve_calls += dual_solve_calls

            if config.profile:
                print(
                    " | ".join(
                        [
                            f"n_features={n_features}",
                            f"rff={rff_time:.3f}s",
                            f"window={window_build_time:.3f}s",
                            f"solve={solve_time:.3f}s",
                            f"total={feature_total_time:.3f}s",
                            f"chunks={chunks_processed}",
                            f"primal_calls={primal_solve_calls}",
                            f"dual_calls={dual_solve_calls}",
                        ]
                    )
                )
    finally:
        if config.deterministic:
            torch.use_deterministic_algorithms(deterministic_prev)

    total_time = time.perf_counter() - run_start

    return FitCoreMetrics(
        device=str(config.device),
        dtype=str(config.dtype),
        n_samples=int(x_lagged.shape[0]),
        n_predictors=int(x_lagged.shape[1]),
        total_features=len(feature_counts),
        total_windows_processed=total_windows_processed,
        total_chunks_processed=total_chunks_processed,
        total_solve_calls=total_solve_calls,
        total_primal_solve_calls=total_primal_solve_calls,
        total_dual_solve_calls=total_dual_solve_calls,
        total_time_s=total_time,
        checksum=checksum,
        feature_profiles=feature_profiles,
    )


def prediction_subset_gpu(
    x_lagged: np.ndarray,
    y_aligned: np.ndarray,
    *,
    window: int,
    n_features: int,
    gamma: float,
    alpha: float,
    seed: int,
    device: torch.device,
    solver_policy: SolverPolicy = "auto",
    dtype: torch.dtype = torch.float32,
    sample_count: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    x_t = torch.tensor(x_lagged, device=device, dtype=dtype)
    y_t = torch.tensor(y_aligned, device=device, dtype=dtype)

    w_t, b_t = generate_rff_weights(
        n_features,
        x_t.shape[1],
        gamma=gamma,
        seed=seed,
        device=device,
        dtype=dtype,
    )
    z_all = rff_transform(x_t, w_t, b_t)

    z_windows, y_windows = build_window_tensors(z_all, y_t, window)
    solver = _select_solver(solver_policy, n_features, window)
    if solver == "primal":
        betas = _solve_primal(z_windows, y_windows, [alpha])
    else:
        betas = _solve_dual(z_windows, y_windows, [alpha], chunk_size_features=None)
    beta = betas[float(alpha)]

    z_pred = z_all[window - 1 : -1]
    y_pred = (beta * z_pred).sum(dim=1)
    y_true = y_t[window:]

    take = min(sample_count, y_pred.shape[0])
    return (
        y_pred[:take].detach().cpu().numpy(),
        y_true[:take].detach().cpu().numpy(),
    )

def print_metrics(metrics: FitCoreMetrics, *, profile: bool) -> None:
    print(f"Device: {metrics.device}")
    print(f"DType: {metrics.dtype}")
    print(f"Samples: {metrics.n_samples}")
    print(f"Predictors: {metrics.n_predictors}")
    print(f"Feature configs: {metrics.total_features}")
    print(f"Total windows processed: {metrics.total_windows_processed}")
    print(f"Total chunks processed: {metrics.total_chunks_processed}")
    print(f"Total solve calls: {metrics.total_solve_calls}")
    print(f"Primal solve calls: {metrics.total_primal_solve_calls}")
    print(f"Dual solve calls: {metrics.total_dual_solve_calls}")
    print(f"Checksum: {metrics.checksum:.6f}")
    print(f"Total time: {metrics.total_time_s:.3f}s")

    if profile and metrics.feature_profiles:
        fastest = min(metrics.feature_profiles, key=lambda x: x.total_time_s)
        slowest = max(metrics.feature_profiles, key=lambda x: x.total_time_s)
        print(
            f"Fastest n_features={fastest.n_features} total={fastest.total_time_s:.3f}s"
        )
        print(
            f"Slowest n_features={slowest.n_features} total={slowest.total_time_s:.3f}s"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CUDA fit-core benchmark for RFF ridge."
    )
    parser.add_argument(
        "--fit-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run fit-core only; prediction row output is disabled in this phase.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable per-feature timing and memory profiling output.",
    )
    parser.add_argument(
        "--chunk-size-windows",
        type=int,
        default=128,
        help="Number of rolling windows per chunk for batched solve.",
    )
    parser.add_argument(
        "--chunk-size-features",
        type=int,
        default=None,
        help="Optional feature chunking size for dual beta materialization.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--solver-policy",
        choices=["auto", "primal", "dual"],
        default="auto",
        help="Ridge solve policy.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=MAX_RFF_FEATURES,
        help="Maximum RFF features.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Enable deterministic algorithms for parity runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    x_lagged, y_aligned, _dates = prepare_dataset()

    device = resolve_device(args.device)

    config = GPUFitConfig(
        windows=TRAINING_WINDOWS,
        max_features=args.max_features,
        gamma=GAMMA,
        alphas=RIDGE_ALPHAS,
        seed=RANDOM_SEED,
        dtype=torch.float32,
        device=device,
        solver_policy=args.solver_policy,
        fit_only=args.fit_only,
        chunk_size_windows=args.chunk_size_windows,
        chunk_size_features=args.chunk_size_features,
        profile=args.profile,
        deterministic=args.deterministic,
    )

    metrics = run_fit_core_gpu(x_lagged, y_aligned, config)
    print_metrics(metrics, profile=args.profile)


if __name__ == "__main__":
    main()
