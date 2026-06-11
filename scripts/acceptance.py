"""Acceptance check: do the paper's qualitative signatures appear?

Reads the aggregated metrics parquets and prints, per window, at the
least-shrunk lambda (most extreme curves): the R2 dip near c = 1 with
recovery in the tail, and the annualized Sharpe rising with complexity.
Also writes the paper-style figures.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from voc.plots import plot_window_metrics


def main() -> None:
    for window in (12, 60, 120):
        path = Path(f"artifacts/metrics_T{window}.parquet")
        if not path.exists():
            print(f"T={window}: missing {path}")
            continue
        frame = pd.read_parquet(path)
        full = frame[frame["subsample"] == "full"]
        lo = full[full["lam"] == 0.001].sort_values("p")
        c = lo["p"] / window
        near_one = lo[(c > 0.5) & (c < 2.0)]
        tail = lo[c > 50]
        head = lo[c <= 0.5]

        r2_dip = near_one["r2_mean"].min()
        r2_tail = tail["r2_mean"].mean()
        sr_head = head["sr_mean"].mean() * math.sqrt(12)
        sr_near = near_one["sr_mean"].min() * math.sqrt(12)
        sr_tail = tail["sr_mean"].mean() * math.sqrt(12)
        line = (
            f"T={window:3d} lam=0.001: R2 dip(c~1)={r2_dip:8.3f} -> tail {r2_tail:7.4f} | "
            f"SR ann: low-c {sr_head:5.2f}, c~1 {sr_near:5.2f}, high-c {sr_tail:5.2f}"
        )
        if "bnrm_mean" in lo.columns:
            spike = near_one["bnrm_mean"].max() / max(tail["bnrm_mean"].mean(), 1e-12)
            line += f" | bnrm spike(c~1)/tail = {spike:8.1f}"
        print(line)

        paths = plot_window_metrics(frame, out_dir=Path("artifacts/figures"))
        print(f"   figures: {', '.join(p.name for p in paths)}")


if __name__ == "__main__":
    main()
