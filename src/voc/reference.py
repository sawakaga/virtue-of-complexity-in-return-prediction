"""Slow float64 oracle: transliteration of the MATLAB rolling fit loop.

Used ONLY by tests. When porting numerics, the dumbest possible
re-implementation of the reference is written first, then the fast clever
version is verified against it — a fast implementation verified only
against itself proves nothing.

Semantics per tryrff_v2_function_for_each_sim.m (demean=0, trainfrq=1):
window k trains on rows [k, k+T), standardizes features by the training
window's ddof=1 std (applied to the test row too), and predicts y[k+T]
from z[k+T]. Ridge effective penalty is lam * T in both MATLAB branches:
ridgesvd is called with lamlist*trnwin, and get_beta divides the Gram by
T. The solve below uses the DUAL form beta = Z'(ZZ' + lam*T*I)^{-1} y;
the primal-form equivalence (push-through identity) is exercised by
test_oracle_matches_primal_closed_form.
"""

from __future__ import annotations

import numpy as np

from voc.rff import project


def run_reference(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int,
    p_grid: list[int],
    lambdas: list[float],
    weights: np.ndarray,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OOS predictions and squared beta norms, [K, nP, nL].

    Prediction row k corresponds to target index window + k.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    n_windows = n - window
    yprd = np.full((n_windows, len(p_grid), len(lambdas)), np.nan)
    bnrm = np.full_like(yprd, np.nan)

    g_full = project(x, weights, gamma=gamma)

    for pi, p in enumerate(p_grid):
        half = p // 2
        g = g_full[:, :half]
        z = np.concatenate([np.cos(g), np.sin(g)], axis=1)

        for k in range(n_windows):
            ztrn = z[k : k + window]
            ztst = z[k + window]
            ytrn = y[k : k + window]

            std = ztrn.std(axis=0, ddof=1)
            ztrn = ztrn / std
            ztst = ztst / std

            gram = ztrn @ ztrn.T
            for li, lam in enumerate(lambdas):
                a = np.linalg.solve(gram + lam * window * np.eye(window), ytrn)
                beta = ztrn.T @ a
                yprd[k, pi, li] = ztst @ beta
                bnrm[k, pi, li] = float(beta @ beta)

    return yprd, bnrm
