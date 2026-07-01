"""Random Fourier Feature weights and projection.

Z(x) = [cos(gamma * W @ x); sin(gamma * W @ x)], W ~ N(0, I), following
tryrff_v2_function_for_each_sim.m:111-120. By Bochner's theorem this
feature map approximates a Gaussian kernel; the number of features is the
complexity dial the paper sweeps.

Weights are drawn ONCE at the maximum half-count and prefix-sliced per
grid value, so the P grid forms a nested family of models. NumPy's
Generator fills row-major, which makes prefix slicing equivalent to a
smaller draw — pinned by test_draw_weights_nested_prefix_property.

The projection G = gamma * X @ W^T is the only O(N*d*H) operation in the
pipeline; computed once per seed, every grid point's features are just
cos/sin of column slices of G. (MATLAB's rng(s) randn stream cannot be
reproduced bit-for-bit from NumPy, so cross-implementation parity tests
inject W explicitly instead.)
"""

from __future__ import annotations

import numpy as np


def draw_weights(*, seed: int, n_inputs: int, max_half: int) -> np.ndarray:
    """Gaussian RFF weights [max_half, n_inputs], deterministic per seed."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((max_half, n_inputs))


def project(x: np.ndarray, w: np.ndarray, *, gamma: float) -> np.ndarray:
    """Scaled projection G = gamma * X @ W^T, shape [N, max_half].

    cos(G[:, a:b]) and sin(G[:, a:b]) are the feature block for half-index
    range [a, b); cos and sin of the SAME projection column belong to the
    same random draw, matching the MATLAB [cos(...); sin(...)] stack.
    """
    return gamma * (x @ w.T)
