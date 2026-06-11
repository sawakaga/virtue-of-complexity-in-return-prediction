"""Fast rolling RFF-ridge fit core (torch, device-agnostic).

Mathematically identical to reference.py (proven by test_solver parity);
engineered around three observations:

1.  The within-window standardized Gram K = Z diag(1/S^2) Z' is a SUM of
    per-feature outer products, so when the grid moves from P1 to P2 only
    the new features' contribution needs computing. Summed over the whole
    grid the matmul cost drops from K*T^2*sum(P) to K*T^2*max(P) — each
    feature enters exactly once (~12x fewer flops for T=120).

2.  K + lam*T*I shares eigenvectors for every lam — only the eigenvalues
    shift. One batched eigh per (window-chunk, grid point) followed by a
    diagonal rescale per lambda replaces seven factorizations. eigh is the
    right factorization here: symmetric PSD input, cheaper than SVD, and
    it does not fail outright when an eigenvalue underflows near the
    interpolation threshold P ~ T (Cholesky would).

3.  Everything stays in T-space (T <= 120) via the dual/representer form:
    beta = Z'a never needs materializing. Predictions need only
    ktst = Ztrn diag(1/S^2) ztst (a T-vector) and the squared norm comes
    free from the eigen-coefficients:
        ||beta||^2 = a'Ka = sum_i d_i * (u_i'y / (d_i + lam*T))^2.

Outputs accumulate on-device and transfer once per call — per-element
.item() reads would force a host-device sync each time on CUDA/MPS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from voc.rff import project


@dataclass(slots=True)
class SeedRunResult:
    """OOS predictions and squared beta norms for one (seed, window) run.

    Row k of each array corresponds to target index window + k in the
    input sample (prediction date = dates[window + k]).
    """

    yprd: np.ndarray  # [K, nP, nL]
    bnrm: np.ndarray  # [K, nP, nL]


def fit_one_seed(
    x: np.ndarray,
    y: np.ndarray,
    *,
    window: int,
    p_grid: list[int],
    lambdas: list[float],
    weights: np.ndarray,
    gamma: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
    chunk_windows: int = 128,
    chunk_half_features: int = 1024,
) -> SeedRunResult:
    """Rolling ridge over all grid points and lambdas for one weight draw.

    Window k trains on rows [k, k+window) and predicts y[k+window] from
    z[k+window] — the index convention pinned by the synthetic alignment
    test (this fixes the old implementations' stale-feature bug).

    chunk_windows bounds the [C, T, T] Gram batch; chunk_half_features
    bounds the materialized feature block [C, T, 2*B] (the unfold view
    becomes a copy once divided by the per-window std).
    """
    if window < 2:
        raise ValueError("window must be >= 2 for a ddof=1 std")
    halves = [p // 2 for p in p_grid]
    if halves != sorted(halves):
        raise ValueError("p_grid must be non-decreasing for incremental updates")

    device = device or torch.device("cpu")
    n = y.shape[0]
    n_windows = n - window
    if n_windows < 1:
        raise ValueError("sample shorter than training window")

    max_half = halves[-1]
    # The projection is the only O(N*d*H) op; everything downstream uses
    # cos/sin of column slices of g.
    g = torch.as_tensor(project(x, weights[:max_half], gamma=gamma), dtype=dtype, device=device)
    y_t = torch.as_tensor(np.asarray(y, dtype=np.float64), dtype=dtype, device=device)

    # Effective ridge penalty lam * T (MATLAB lamlist*trnwin semantics).
    lam_t = torch.as_tensor(lambdas, dtype=dtype, device=device) * window

    n_p, n_l = len(p_grid), len(lambdas)
    yprd = torch.empty((n_windows, n_p, n_l), dtype=dtype, device=device)
    bnrm = torch.empty_like(yprd)

    # y unfolded into training windows: row k holds y[k .. k+T-1].
    y_windows = y_t.unfold(0, window, 1)[:n_windows]

    for c0 in range(0, n_windows, chunk_windows):
        c1 = min(c0 + chunk_windows, n_windows)
        chunk = c1 - c0

        kacc = torch.zeros((chunk, window, window), dtype=dtype, device=device)
        ktst = torch.zeros((chunk, window), dtype=dtype, device=device)
        y_c = y_windows[c0:c1]

        prev_half = 0
        for pi, half in enumerate(halves):
            # Accumulate only the NEW features' contribution (block-additive
            # because the per-window std weights features independently).
            for b0 in range(prev_half, half, chunk_half_features):
                b1 = min(b0 + chunk_half_features, half)
                _accumulate_block(g[:, b0:b1], kacc, ktst, c0=c0, c1=c1, window=window)
            prev_half = half

            evals, u = torch.linalg.eigh(kacc)
            # u'y once; every lambda is then a diagonal rescale.
            uty = (u.transpose(1, 2) @ y_c.unsqueeze(-1)).squeeze(-1)  # [C, T]
            coef = uty.unsqueeze(-1) / (evals.unsqueeze(-1) + lam_t.view(1, 1, n_l))
            a = u @ coef  # [C, T, L] dual solutions for all lambdas
            yprd[c0:c1, pi, :] = torch.einsum("ctl,ct->cl", a, ktst)
            bnrm[c0:c1, pi, :] = (evals.unsqueeze(-1) * coef * coef).sum(dim=1)

    return SeedRunResult(
        yprd=yprd.cpu().numpy(),
        bnrm=bnrm.cpu().numpy(),
    )


def _accumulate_block(
    g_cols: torch.Tensor,
    kacc: torch.Tensor,
    ktst: torch.Tensor,
    *,
    c0: int,
    c1: int,
    window: int,
) -> None:
    """Add one cos/sin feature block to the standardized Gram accumulators.

    For each window k in [c0, c1): scale the block's train rows and the
    test row by the TRAIN window's ddof=1 std (per feature), then add
    Ztrn~ @ Ztrn~' to kacc[k] and Ztrn~ @ ztst~ to ktst[k].
    """
    z_block = torch.cat([torch.cos(g_cols), torch.sin(g_cols)], dim=1)  # [N, 2B]
    n_feat = z_block.shape[1]

    # Per-window train std via prefix sums: O(N) per feature instead of
    # O(K*T). The E[x^2]-E[x]^2 form risks cancellation in fp32, but
    # cos/sin are bounded in [-1, 1] and the parity tests run fp64.
    zero = z_block.new_zeros((1, n_feat))
    s1 = torch.cat([zero, torch.cumsum(z_block, dim=0)])
    s2 = torch.cat([zero, torch.cumsum(z_block * z_block, dim=0)])
    sums = s1[c0 + window : c1 + window] - s1[c0:c1]  # [C, 2B]
    sqs = s2[c0 + window : c1 + window] - s2[c0:c1]
    var = (sqs - sums * sums / window) / (window - 1)
    std = var.clamp_min(0).sqrt()

    # unfold creates a strided VIEW [K_all, 2B, T]; slicing the chunk and
    # dividing by std materializes only [C, T, 2B].
    ztrn = z_block.unfold(0, window, 1)[c0:c1].transpose(1, 2) / std.unsqueeze(1)
    ztst = z_block[c0 + window : c1 + window] / std  # [C, 2B]

    kacc += ztrn @ ztrn.transpose(1, 2)
    ktst += (ztrn @ ztst.unsqueeze(-1)).squeeze(-1)
