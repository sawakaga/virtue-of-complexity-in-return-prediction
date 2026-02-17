from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def load_thesis_gpu_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src" / "us-100years" / "thesis_gpu.py"
    spec = importlib.util.spec_from_file_location("thesis_gpu", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(module, **kwargs):
    defaults = {
        "windows": [12],
        "max_features": 12,
        "gamma": 2.0,
        "alphas": [1e-1, 1.0],
        "seed": 123,
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "solver_policy": "auto",
        "fit_only": True,
        "chunk_size_windows": 32,
        "chunk_size_features": None,
        "profile": False,
        "deterministic": False,
    }
    defaults.update(kwargs)
    return module.GPUFitConfig(**defaults)


def test_build_window_tensors_shapes_for_required_windows():
    module = load_thesis_gpu_module()

    n = 140
    f = 16
    z = torch.randn(n, f)
    y = torch.randn(n)

    for window in [12, 60, 120]:
        z_windows, y_windows = module.build_window_tensors(z, y, window)
        assert z_windows.shape == (n - window, window, f)
        assert y_windows.shape == (n - window, window)


def test_fit_core_deterministic_with_same_seed():
    module = load_thesis_gpu_module()

    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 6)).astype(np.float32)
    y = rng.normal(size=(80,)).astype(np.float32)

    config = _config(module, windows=[10], max_features=10, seed=777, chunk_size_windows=16)

    m1 = module.run_fit_core_gpu(x, y, config)
    m2 = module.run_fit_core_gpu(x, y, config)

    assert m1.total_solve_calls == m2.total_solve_calls
    assert m1.total_chunks_processed == m2.total_chunks_processed
    assert np.isclose(m1.checksum, m2.checksum, rtol=0, atol=1e-7)


def test_solver_policy_auto_primal_when_f_le_t():
    module = load_thesis_gpu_module()

    rng = np.random.default_rng(11)
    x = rng.normal(size=(90, 8)).astype(np.float32)
    y = rng.normal(size=(90,)).astype(np.float32)

    # window=12 and only n_features=12 => primal branch.
    config = _config(module, windows=[12], max_features=12, solver_policy="auto")
    metrics = module.run_fit_core_gpu(x, y, config)

    assert metrics.total_primal_solve_calls == metrics.total_solve_calls
    assert metrics.total_dual_solve_calls == 0


def test_solver_policy_auto_dual_when_f_gt_t():
    module = load_thesis_gpu_module()

    rng = np.random.default_rng(12)
    x = rng.normal(size=(90, 8)).astype(np.float32)
    y = rng.normal(size=(90,)).astype(np.float32)

    # window=12 and only n_features=24 => dual branch.
    config = _config(module, windows=[12], max_features=24, solver_policy="auto")
    metrics = module.run_fit_core_gpu(x, y, config)

    assert metrics.total_dual_solve_calls > 0


def test_tiny_cpu_gpu_parity_auto_and_dual_policy():
    module = load_thesis_gpu_module()

    rng = np.random.default_rng(42)
    x = rng.normal(size=(60, 8)).astype(np.float32)
    y = rng.normal(size=(60,)).astype(np.float32)

    auto_config = _config(
        module,
        windows=[10],
        max_features=20,
        alphas=[1e-1, 1.0],
        chunk_size_windows=20,
        solver_policy="auto",
    )
    dual_config = _config(
        module,
        windows=[10],
        max_features=20,
        alphas=[1e-1, 1.0],
        chunk_size_windows=20,
        solver_policy="dual",
    )

    auto_gpu = module.run_fit_core_gpu(x, y, auto_config)
    auto_cpu = module.run_fit_core_cpu_reference(x, y, auto_config)
    dual_gpu = module.run_fit_core_gpu(x, y, dual_config)
    dual_cpu = module.run_fit_core_cpu_reference(x, y, dual_config)

    assert auto_gpu.total_solve_calls == auto_cpu.total_solve_calls
    assert np.isclose(auto_gpu.checksum, auto_cpu.checksum, rtol=1e-4, atol=1e-5)

    assert dual_gpu.total_solve_calls == dual_cpu.total_solve_calls
    assert np.isclose(dual_gpu.checksum, dual_cpu.checksum, rtol=1e-4, atol=1e-5)


def test_real_data_slice_cpu_gpu_parity():
    module = load_thesis_gpu_module()

    x, y, _dates = module.prepare_dataset()
    x = x[:200]
    y = y[:200]

    config = _config(module, windows=[12], max_features=24, alphas=[1e-1, 1.0], chunk_size_windows=32)

    gpu_metrics = module.run_fit_core_gpu(x, y, config)
    cpu_metrics = module.run_fit_core_cpu_reference(x, y, config)

    assert gpu_metrics.total_solve_calls == cpu_metrics.total_solve_calls
    assert np.isclose(gpu_metrics.checksum, cpu_metrics.checksum, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_tiny_parity_and_determinism():
    module = load_thesis_gpu_module()

    rng = np.random.default_rng(1234)
    x = rng.normal(size=(72, 10)).astype(np.float32)
    y = rng.normal(size=(72,)).astype(np.float32)

    config = _config(
        module,
        windows=[12],
        max_features=24,
        alphas=[1e-1, 1.0],
        device=torch.device("cuda"),
        solver_policy="auto",
        deterministic=True,
    )

    m1 = module.run_fit_core_gpu(x, y, config)
    m2 = module.run_fit_core_gpu(x, y, config)
    cpu_ref = module.run_fit_core_cpu_reference(
        x,
        y,
        _config(
            module,
            windows=[12],
            max_features=24,
            alphas=[1e-1, 1.0],
            device=torch.device("cpu"),
            solver_policy="auto",
        ),
    )

    assert np.isclose(m1.checksum, m2.checksum, rtol=0, atol=1e-6)
    assert np.isclose(m1.checksum, cpu_ref.checksum, rtol=1e-4, atol=1e-5)
