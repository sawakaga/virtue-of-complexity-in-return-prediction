"""Device resolution and dtype policy.

Hardware facts driving the policy:
- Apple GPUs (MPS) have no fp64 ALUs; torch raises on float64 MPS
  tensors, so MPS defaults to fp32. Whether fp32 GEMMs on the GPU beat
  Accelerate-backed fp64 GEMMs on the CPU is an empirical question on
  unified memory — `make bench` answers it; nothing here assumes it.
- Metal's linalg coverage (eigh in particular) is incomplete across torch
  versions; the solver routes factorizations through the CPU when the
  device lacks them (see solver._eigh_anywhere).
- CPU and CUDA default to fp64: near the interpolation threshold P ~ T
  the smallest Gram eigenvalues approach zero, and with lam = 1e-3 * T
  the resolvent 1/(eval + lam*T) amplifies eigenvalue error — fp32 there
  visibly distorts the double-descent spike.
"""

from __future__ import annotations

import torch


def resolve_device(name: str) -> torch.device:
    """Map a CLI device name to a usable torch.device.

    "auto" prefers CUDA, then MPS, then CPU.
    """
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available on this host")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available on this host")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device name: {name!r}")


def default_dtype(device: torch.device) -> torch.dtype:
    """fp32 on MPS (no fp64 hardware), fp64 everywhere else."""
    return torch.float32 if device.type == "mps" else torch.float64
