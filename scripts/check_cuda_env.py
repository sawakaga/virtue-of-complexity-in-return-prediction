from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"python_version={sys.version.split()[0]}")
    print(f"platform={platform.platform()}")

    try:
        import torch
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"torch_import_error={exc}")
        return 2

    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_compiled={torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"torch_cuda_available={cuda_available}")

    if not cuda_available:
        print("status=cpu_only_or_missing_cuda")
        print("hint=Install CUDA-enabled PyTorch build and verify NVIDIA driver.")
        return 1

    device_count = torch.cuda.device_count()
    print(f"cuda_device_count={device_count}")

    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        print(
            f"cuda_device_{idx}=name:{name},cc:{props.major}.{props.minor},"
            f"total_mem_gb:{total_gb:.2f}"
        )

    print("status=cuda_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
