"""Tests for the paper-figure plotting helpers."""

import numpy as np
import pandas as pd

from voc.plots import plot_window_metrics

RNG = np.random.default_rng(99)


def _fake_metrics(window=12, n_p=6, lambdas=(0.001, 1.0, 1000.0)):
    p_values = [2, 6, 12, 24, 120, 360][:n_p]
    rows = []
    for sub in ("full", "1975-2020"):
        for p in p_values:
            for lam in lambdas:
                rows.append(
                    {
                        "window": window,
                        "subsample": sub,
                        "p": p,
                        "lam": lam,
                        "n_seeds": 3,
                        "r2_mean": RNG.normal(scale=0.1),
                        "sr_mean": RNG.normal(loc=0.1, scale=0.05),
                        "bnrm_mean": abs(RNG.normal()),
                        "alpha_t_mean": RNG.normal(),
                    }
                )
    return pd.DataFrame(rows)


def test_plot_window_metrics_writes_figures(tmp_path):
    frame = _fake_metrics()

    paths = plot_window_metrics(frame, out_dir=tmp_path, subsample="full")

    assert len(paths) >= 2
    for path in paths:
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 1000
    names = {p.name for p in paths}
    assert any("r2" in n for n in names)
    assert any("sr" in n for n in names)
