"""Paper-style figures from aggregated metrics parquet files.

X axis is model complexity c = P/T on a log scale (the paper's
convention): the interesting region spans four orders of magnitude and
the double-descent spike at c = 1 would be invisible on a linear axis.
One line per lambda; Sharpe is annualized here (sqrt(12)) and nowhere
else — metrics stay monthly on disk.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never require a display server
import matplotlib.pyplot as plt
import pandas as pd

_PANELS: list[tuple[str, str, float]] = [
    # (metric column, axis label, annualization factor)
    ("r2_mean", "OOS R² (monthly)", 1.0),
    ("sr_mean", "Sharpe ratio (annualized)", math.sqrt(12.0)),
    ("bnrm_mean", "||beta||² (mean)", 1.0),
    ("alpha_t_mean", "alpha t-statistic", 1.0),
]


def plot_window_metrics(
    frame: pd.DataFrame,
    *,
    out_dir: Path,
    subsample: str = "full",
) -> list[Path]:
    """One figure per metric: metric vs c = P/T, one line per lambda.

    Returns the written file paths. Missing metric columns are skipped so
    partial frames (e.g. smoke runs) still plot.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sub = frame[frame["subsample"] == subsample]
    window = int(sub["window"].iloc[0])
    written: list[Path] = []

    for column, label, scale in _PANELS:
        if column not in sub.columns:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for lam, group in sub.groupby("lam"):
            group = group.sort_values("p")
            ax.plot(
                group["p"] / window,
                group[column] * scale,
                marker=".",
                markersize=3,
                linewidth=1.0,
                label=f"λ={lam:g}",
            )
        ax.axvline(1.0, color="grey", linestyle=":", linewidth=1)  # c = 1
        ax.set_xscale("log")
        ax.set_xlabel("complexity c = P / T")
        ax.set_ylabel(label)
        ax.set_title(f"T={window}, subsample={subsample}")
        ax.legend(fontsize=8)
        fig.tight_layout()

        path = out_dir / f"{column.removesuffix('_mean')}_T{window}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written
