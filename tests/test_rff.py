"""Tests for Random Fourier Feature weight generation and projection.

The nested-prefix property is load-bearing: the paper draws W once at max
P and slices prefixes, so the P=200 model contains the P=100 model's
features exactly. Our incremental-Gram optimization relies on the same
property, so it is pinned here independently of the solver.
"""

import numpy as np

from voc.rff import draw_weights, project


def test_draw_weights_shape_and_determinism():
    w1 = draw_weights(seed=3, n_inputs=15, max_half=500)
    w2 = draw_weights(seed=3, n_inputs=15, max_half=500)
    w3 = draw_weights(seed=4, n_inputs=15, max_half=500)

    assert w1.shape == (500, 15)
    assert w1.dtype == np.float64
    np.testing.assert_array_equal(w1, w2)
    assert not np.array_equal(w1, w3)


def test_draw_weights_nested_prefix_property():
    # Drawing fewer rows with the same seed must reproduce the prefix of a
    # larger draw — this is what makes the P grid a nested model family.
    big = draw_weights(seed=11, n_inputs=4, max_half=300)
    small = draw_weights(seed=11, n_inputs=4, max_half=120)
    np.testing.assert_array_equal(big[:120], small)


def test_draw_weights_are_standard_normal():
    w = draw_weights(seed=0, n_inputs=10, max_half=6000)
    assert abs(w.mean()) < 0.01
    assert abs(w.std() - 1.0) < 0.01


def test_project_applies_gamma_scaling():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(7, 3))
    w = draw_weights(seed=1, n_inputs=3, max_half=20)

    g = project(x, w, gamma=2.0)

    assert g.shape == (7, 20)
    np.testing.assert_allclose(g, 2.0 * x @ w.T, rtol=1e-12)
