"""Tests for device resolution and the MPS execution path.

Apple GPUs have no fp64 ALUs (torch raises on float64 MPS tensors) and
Metal's linalg coverage is incomplete, so the policy is: GEMM-heavy work
may run on MPS in fp32, factorizations fall back to CPU when the device
lacks them, and CPU/CUDA default to fp64.
"""

import numpy as np
import pytest
import torch

from voc.device import default_dtype, resolve_device
from voc.rff import draw_weights
from voc.solver import fit_one_seed

MPS_AVAILABLE = torch.backends.mps.is_available()


def test_resolve_explicit_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_auto_returns_usable_device():
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    torch.zeros(1, device=device)  # must be constructible


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_default_dtype_policy():
    assert default_dtype(torch.device("cpu")) == torch.float64
    assert default_dtype(torch.device("mps")) == torch.float32


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_resolve_mps_and_auto_prefers_accelerator():
    assert resolve_device("mps").type == "mps"
    if not torch.cuda.is_available():
        assert resolve_device("auto").type == "mps"


@pytest.mark.skipif(not MPS_AVAILABLE, reason="MPS not available")
def test_solver_on_mps_matches_cpu_fp32():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 4))
    y = rng.normal(size=80)
    w = draw_weights(seed=5, n_inputs=4, max_half=30)

    cpu = fit_one_seed(
        x,
        y,
        window=12,
        p_grid=[14, 60],
        lambdas=[0.1, 10.0],
        weights=w,
        gamma=2.0,
        dtype=torch.float32,
    )
    mps = fit_one_seed(
        x,
        y,
        window=12,
        p_grid=[14, 60],
        lambdas=[0.1, 10.0],
        weights=w,
        gamma=2.0,
        device=torch.device("mps"),
        dtype=torch.float32,
    )

    np.testing.assert_allclose(mps.yprd, cpu.yprd, atol=2e-3)
    np.testing.assert_allclose(mps.bnrm, cpu.bnrm, rtol=2e-2, atol=1e-3)
