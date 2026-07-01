"""Benchmark the fit core: CPU fp64 vs MPS fp32 on the real workload.

Run with: make bench  (or: uv run python scripts/bench.py [--windows ...])

Medians of repeated runs, never single timings: first-run JIT/cache warmup
and laptop thermal throttling both lie. The MPS path keeps GEMMs on the
GPU and routes eigh through the CPU (Metal lacks it), so this benchmark
answers whether that hybrid beats Accelerate-backed fp64 outright.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from voc.constants import GAMMA, LAMBDA_GRID, MAX_P
from voc.grids import plist
from voc.preprocess import prepare_dataset
from voc.rff import draw_weights
from voc.solver import fit_one_seed


def time_config(prep, window, device, dtype, repeats=3) -> list[float]:
    grid = plist(window)
    weights = draw_weights(seed=1, n_inputs=prep.x.shape[1], max_half=MAX_P // 2)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fit_one_seed(
            prep.x,
            prep.y,
            window=window,
            p_grid=grid,
            lambdas=LAMBDA_GRID,
            weights=weights,
            gamma=GAMMA,
            device=device,
            dtype=dtype,
        )
        times.append(time.perf_counter() - start)
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=int, nargs="+", default=[12, 60, 120])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    prep = prepare_dataset()
    configs: list[tuple[str, torch.device, torch.dtype]] = [
        ("cpu/fp64", torch.device("cpu"), torch.float64),
        ("cpu/fp32", torch.device("cpu"), torch.float32),
    ]
    if torch.backends.mps.is_available():
        configs.append(("mps/fp32", torch.device("mps"), torch.float32))
    if torch.cuda.is_available():
        configs.append(("cuda/fp32", torch.device("cuda"), torch.float32))
        configs.append(("cuda/fp64", torch.device("cuda"), torch.float64))

    print(f"sample N={len(prep.y)}, repeats={args.repeats} (reporting median)")
    print(f"{'window':>8} | " + " | ".join(f"{name:>10}" for name, _, _ in configs))
    for window in args.windows:
        cells = []
        for _, device, dtype in configs:
            runs = time_config(prep, window, device, dtype, repeats=args.repeats)
            cells.append(f"{statistics.median(runs):9.2f}s")
        print(f"{window:>8} | " + " | ".join(f"{c:>10}" for c in cells))


if __name__ == "__main__":
    main()
