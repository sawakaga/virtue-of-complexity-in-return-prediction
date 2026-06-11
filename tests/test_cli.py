"""Tests for the multi-seed CLI driver."""

import numpy as np
import pandas as pd

from voc.cli import main, parse_args
from voc.constants import LAMBDA_GRID, TRAINING_WINDOWS
from voc.grids import plist


def test_parse_args_defaults():
    args = parse_args([])

    assert args.windows == TRAINING_WINDOWS
    assert args.n_seeds == 10
    assert args.first_seed == 1
    assert args.device == "auto"
    assert args.dtype == "float64"
    assert not args.save_predictions


def test_plist_clips_to_small_max_p_for_smoke_runs():
    # The MATLAB formula assumes maxP >= 31*T; for smaller smoke-test
    # budgets the grid must stay sorted and bounded by max_p.
    grid = plist(12, max_p=120)

    assert grid == sorted(grid)
    assert grid[-1] == 120
    assert all(p <= 120 for p in grid)
    assert 2 in grid and 12 in grid and 16 in grid


def test_main_smoke_run_writes_metrics_parquet(tmp_path):
    main(
        [
            "--windows",
            "12",
            "--n-seeds",
            "2",
            "--max-p",
            "120",
            "--out-dir",
            str(tmp_path),
        ]
    )

    out = tmp_path / "metrics_T12.parquet"
    assert out.exists()
    frame = pd.read_parquet(out)

    grid = plist(12, max_p=120)
    subsamples = frame["subsample"].unique()
    assert len(frame) == len(grid) * len(LAMBDA_GRID) * len(subsamples)
    assert "full" in subsamples and "1975-2020" in subsamples
    assert (frame["n_seeds"] == 2).all()
    assert {"sr_mean", "r2_mean", "alpha_t_mean", "sr_p2_5", "window"} <= set(frame.columns)
    assert (frame["window"] == 12).all()
    assert np.isfinite(frame["sr_mean"]).all()


def test_main_save_predictions_writes_per_seed_parquet(tmp_path):
    main(
        [
            "--windows",
            "12",
            "--n-seeds",
            "1",
            "--max-p",
            "24",
            "--out-dir",
            str(tmp_path),
            "--save-predictions",
        ]
    )

    pred_file = tmp_path / "predictions" / "T12_seed1.parquet"
    assert pred_file.exists()
    frame = pd.read_parquet(pred_file)
    assert {"date", "p", "lam", "yprd", "bnrm", "y_true"} <= set(frame.columns)
    # K windows x grid x lambdas rows, long format.
    grid = plist(12, max_p=24)
    assert len(frame) % (len(grid) * len(LAMBDA_GRID)) == 0
