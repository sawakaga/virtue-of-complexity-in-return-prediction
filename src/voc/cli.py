"""Multi-seed driver: the Python equivalent of predictions_main.m.

Seeds are 1..n_seeds like MATLAB's iSim. Per (window, seed) the solver
produces the full prediction cube; metrics are computed per seed (the
paper's protocol — metrics first, averaging after, so nonlinear ratios
like Sharpe are means of per-seed values, not metrics of averaged
forecasts) and aggregated into mean + percentile bands.

Output is parquet, not CSV: columnar, typed, compressed — the aggregated
metrics for a full run are a few hundred KB instead of the old 860 MB
prediction dump. Raw per-seed predictions are written only on request
(--save-predictions), sharded per (window, seed) like MATLAB's iSim*.mat.

All metric columns are MONTHLY; annualization (sqrt(12) for Sharpe, x12
for means) is applied at plotting/reporting time only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from voc.constants import GAMMA, LAMBDA_GRID, MAX_P, SUBSAMPLES, TRAINING_WINDOWS
from voc.device import default_dtype, resolve_device
from voc.grids import plist
from voc.metrics import aggregate_seed_metrics, seed_config_metrics
from voc.preprocess import PreparedData, prepare_dataset
from voc.rff import draw_weights
from voc.solver import SeedRunResult, fit_one_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RFF ridge OOS run (Kelly-Malamud-Zhou reproduction)."
    )
    parser.add_argument("--windows", type=int, nargs="+", default=TRAINING_WINDOWS)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--max-p", type=int, default=MAX_P)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float64", "float32"], default="auto")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--chunk-windows", type=int, default=128)
    parser.add_argument("--chunk-half-features", type=int, default=1024)
    return parser.parse_args(argv)


_DTYPES = {"float64": torch.float64, "float32": torch.float32}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = default_dtype(device) if args.dtype == "auto" else _DTYPES[args.dtype]
    print(f"device={device.type} dtype={str(dtype).removeprefix('torch.')}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = prepare_dataset()
    seeds = list(range(args.first_seed, args.first_seed + args.n_seeds))
    subsample_specs: list[tuple[str, tuple[int, int] | None]] = [("full", None)] + [
        (f"{beg}-{end}", (beg, end)) for beg, end in SUBSAMPLES
    ]

    for window in args.windows:
        grid = plist(window, max_p=args.max_p)
        y_oos = prep.y[window:]
        dates_oos = prep.dates[window:]
        per_sub: dict[str, list[dict[str, np.ndarray]]] = {
            label: [] for label, _ in subsample_specs
        }

        for seed in tqdm(seeds, desc=f"T={window}", unit="seed"):
            weights = draw_weights(seed=seed, n_inputs=prep.x.shape[1], max_half=max(grid) // 2)
            result = fit_one_seed(
                prep.x,
                prep.y,
                window=window,
                p_grid=grid,
                lambdas=LAMBDA_GRID,
                weights=weights,
                gamma=GAMMA,
                device=device,
                dtype=dtype,
                chunk_windows=args.chunk_windows,
                chunk_half_features=args.chunk_half_features,
            )
            for label, bounds in subsample_specs:
                per_sub[label].append(
                    seed_config_metrics(result.yprd, y_oos, dates_oos, subsample=bounds)
                )
            if args.save_predictions:
                _write_predictions(out_dir, window, seed, grid, prep, result)

        frames = []
        for label, _ in subsample_specs:
            frame = aggregate_seed_metrics(per_sub[label], p_grid=grid, lambdas=LAMBDA_GRID)
            frame.insert(0, "subsample", label)
            frames.append(frame)
        combined = pd.concat(frames, ignore_index=True)
        combined.insert(0, "window", window)

        path = out_dir / f"metrics_T{window}.parquet"
        combined.to_parquet(path, index=False)
        print(f"T={window}: {len(combined)} metric rows -> {path}")


def _write_predictions(
    out_dir: Path,
    window: int,
    seed: int,
    grid: list[int],
    prep: PreparedData,
    result: SeedRunResult,
) -> None:
    """Long-format per-seed dump, sharded like MATLAB's iSim<seed>.mat."""
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    n_windows, n_p, n_l = result.yprd.shape
    frame = pd.DataFrame(
        {
            "date": np.repeat(prep.dates[window:], n_p * n_l),
            "p": np.tile(np.repeat(grid, n_l), n_windows),
            "lam": np.tile(LAMBDA_GRID, n_windows * n_p),
            "yprd": result.yprd.ravel(),
            "bnrm": result.bnrm.ravel(),
            "y_true": np.repeat(prep.y[window:], n_p * n_l),
        }
    )
    frame.to_parquet(pred_dir / f"T{window}_seed{seed}.parquet", index=False)


if __name__ == "__main__":
    main()
