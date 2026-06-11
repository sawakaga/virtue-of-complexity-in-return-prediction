"""The paper's RFF feature-count grid (tryrff_v2_function_for_each_sim.m:22).

Non-uniform by design: dense around P = T (the interpolation threshold,
where the double-descent spike in OOS performance lives) and sparse in the
smooth high-complexity tail. Replicated verbatim from the MATLAB colon
expression, including any duplicate values, so figures stay point-
comparable with the paper's.
"""

from __future__ import annotations

from voc.constants import MAX_P


def _colon(start: int, step: int, stop: int) -> list[int]:
    """MATLAB a:s:b — inclusive of start, last element <= stop."""
    if step <= 0:
        raise ValueError("step must be positive")
    return list(range(start, stop + 1, step))


def plist(trnwin: int, *, max_p: int = MAX_P) -> list[int]:
    """Grid of total RFF feature counts for a training window length.

    MATLAB: [2, 5:floor(T/10):(T-5), (T-4):2:(T+4),
             (T+5):floor(T/2):30T, 31T:10T:(maxP-1), maxP]
    """
    if trnwin < 10:
        raise ValueError("trnwin must be >= 10 (floor(T/10) step would be 0)")
    return [
        2,
        *_colon(5, trnwin // 10, trnwin - 5),
        *_colon(trnwin - 4, 2, trnwin + 4),
        *_colon(trnwin + 5, trnwin // 2, 30 * trnwin),
        *_colon(31 * trnwin, 10 * trnwin, max_p - 1),
        max_p,
    ]


def feature_count(p: int) -> int:
    """Actual feature dimension for grid value p.

    MATLAB draws floor(p/2) Gaussian weight rows and stacks [cos; sin],
    so odd grid values yield p-1 features.
    """
    return 2 * (p // 2)
