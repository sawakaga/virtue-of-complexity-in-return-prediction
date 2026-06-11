"""Tests for the rolling RFF-ridge fit core.

Strategy: a slow float64 NumPy oracle (reference.py) transliterates the
MATLAB t-loop in DUAL form; the fast torch solver (solver.py) is verified
against it. The oracle itself is verified against the PRIMAL closed form
(Z'Z + lam*T*I)^{-1} Z'y, so the push-through identity connecting the two
MATLAB branches (ridgesvd for P<=T, get_beta for P>T) is crossed by the
test suite rather than assumed.
"""

import numpy as np
import torch

from voc.constants import LAMBDA_GRID
from voc.reference import run_reference
from voc.rff import draw_weights, project
from voc.solver import fit_one_seed

RNG = np.random.default_rng(42)


def _toy_problem(n=90, d=4):
    x = RNG.normal(size=(n, d))
    y = RNG.normal(size=n)
    w = draw_weights(seed=9, n_inputs=d, max_half=40)
    return x, y, w


def test_oracle_matches_primal_closed_form():
    """Oracle (dual, T-space) vs direct primal ridge with lam*T penalty."""
    x, y, w = _toy_problem(n=40)
    window, p = 10, 6  # P <= T regime
    lambdas = [0.01, 1.0, 100.0]

    yprd, bnrm = run_reference(
        x, y, window=window, p_grid=[p], lambdas=lambdas, weights=w, gamma=2.0
    )

    g = project(x, w[: p // 2], gamma=2.0)
    z = np.concatenate([np.cos(g), np.sin(g)], axis=1)
    for k in (0, 7, 29):
        ztrn, ztst, ytrn = z[k : k + window], z[k + window], y[k : k + window]
        s = ztrn.std(axis=0, ddof=1)
        ztrn, ztst = ztrn / s, ztst / s
        for li, lam in enumerate(lambdas):
            beta = np.linalg.solve(ztrn.T @ ztrn + lam * window * np.eye(p), ztrn.T @ ytrn)
            np.testing.assert_allclose(yprd[k, 0, li], ztst @ beta, rtol=1e-9)
            np.testing.assert_allclose(bnrm[k, 0, li], (beta**2).sum(), rtol=1e-9)


def test_solver_matches_oracle_both_regimes():
    """Fast solver vs oracle: P <= T and P > T, all 7 lambdas, Yprd + Bnrm."""
    x, y, w = _toy_problem()
    window = 12
    p_grid = [2, 8, 12, 14, 24, 60]  # crosses the interpolation threshold

    yprd_ref, bnrm_ref = run_reference(
        x, y, window=window, p_grid=p_grid, lambdas=LAMBDA_GRID, weights=w, gamma=2.0
    )
    result = fit_one_seed(
        x, y, window=window, p_grid=p_grid, lambdas=LAMBDA_GRID, weights=w, gamma=2.0
    )

    assert result.yprd.shape == (len(y) - window, len(p_grid), len(LAMBDA_GRID))
    np.testing.assert_allclose(result.yprd, yprd_ref, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(result.bnrm, bnrm_ref, rtol=1e-7, atol=1e-10)


def test_alignment_recovers_perfect_predictability():
    """If y[t] is an exact function of z[t], OOS error must be ~0.

    z is built from white-noise x, so features at t-1 carry no information
    about y[t]: the old off-by-one indexing (predicting y[k+T] from
    z[k+T-1]) produces O(1) errors here and cannot pass.
    """
    n, d, window, p = 120, 3, 20, 6
    x = RNG.normal(size=(n, d))
    w = draw_weights(seed=21, n_inputs=d, max_half=p // 2)

    g = project(x, w, gamma=2.0)
    z = np.concatenate([np.cos(g), np.sin(g)], axis=1)
    beta_true = RNG.normal(size=p)
    y = z @ beta_true

    result = fit_one_seed(x, y, window=window, p_grid=[p], lambdas=[1e-10], weights=w, gamma=2.0)

    errors = np.abs(result.yprd[:, 0, 0] - y[window:])
    assert errors.max() < 1e-5, f"max OOS error {errors.max():.3e} — alignment broken"


def test_incremental_gram_equals_fresh_run():
    """Grid point P computed after incremental updates == P computed alone."""
    x, y, w = _toy_problem()
    window = 12

    chained = fit_one_seed(
        x, y, window=window, p_grid=[8, 24, 60], lambdas=LAMBDA_GRID, weights=w, gamma=2.0
    )
    alone = fit_one_seed(
        x, y, window=window, p_grid=[60], lambdas=LAMBDA_GRID, weights=w, gamma=2.0
    )

    np.testing.assert_allclose(chained.yprd[:, 2, :], alone.yprd[:, 0, :], rtol=1e-9)
    np.testing.assert_allclose(chained.bnrm[:, 2, :], alone.bnrm[:, 0, :], rtol=1e-9)


def test_solver_chunking_invariance():
    """Results must not depend on window/feature chunk sizes."""
    x, y, w = _toy_problem()
    base = fit_one_seed(x, y, window=12, p_grid=[14, 60], lambdas=[0.1, 10.0], weights=w, gamma=2.0)
    chunked = fit_one_seed(
        x,
        y,
        window=12,
        p_grid=[14, 60],
        lambdas=[0.1, 10.0],
        weights=w,
        gamma=2.0,
        chunk_windows=7,
        chunk_half_features=5,
    )
    np.testing.assert_allclose(base.yprd, chunked.yprd, rtol=1e-9)
    np.testing.assert_allclose(base.bnrm, chunked.bnrm, rtol=1e-9)


def test_solver_float32_close_to_float64():
    """fp32 path stays close to fp64 away from the interpolation threshold."""
    x, y, w = _toy_problem()
    f64 = fit_one_seed(x, y, window=12, p_grid=[60], lambdas=[1.0], weights=w, gamma=2.0)
    f32 = fit_one_seed(
        x,
        y,
        window=12,
        p_grid=[60],
        lambdas=[1.0],
        weights=w,
        gamma=2.0,
        dtype=torch.float32,
    )
    np.testing.assert_allclose(f32.yprd, f64.yprd, atol=5e-3)


def test_eigh_falls_back_to_robust_driver_on_lapack_failure(monkeypatch):
    """syevd (divide-and-conquer) can fail code 151 on high-P Grams with
    clustered eigenvalues — observed in real T=120 runs. The solver must
    escalate to the QR-iteration driver and produce the same decomposition.
    """
    real_eigh = torch.linalg.eigh

    def flaky_eigh(matrix, *args, **kwargs):
        raise torch._C._LinAlgError("linalg.eigh: The algorithm failed to converge")

    monkeypatch.setattr(torch.linalg, "eigh", flaky_eigh)
    try:
        x, y, w = _toy_problem()
        fallback = fit_one_seed(
            x, y, window=12, p_grid=[14, 60], lambdas=[0.1, 10.0], weights=w, gamma=2.0
        )
    finally:
        monkeypatch.setattr(torch.linalg, "eigh", real_eigh)

    normal = fit_one_seed(
        x, y, window=12, p_grid=[14, 60], lambdas=[0.1, 10.0], weights=w, gamma=2.0
    )
    np.testing.assert_allclose(fallback.yprd, normal.yprd, rtol=1e-8)
    np.testing.assert_allclose(fallback.bnrm, normal.bnrm, rtol=1e-8)
