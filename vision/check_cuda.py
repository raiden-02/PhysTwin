"""Print whether PyTorch sees CUDA and the selected GPU."""

from __future__ import annotations

import torch


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"cuda_built={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return 1
    print(f"device_count={torch.cuda.device_count()}")
    print(f"device0={torch.cuda.get_device_name(0)}")
    x = torch.zeros(1, device="cuda")
    print(f"alloc_ok={x.device.type == 'cuda'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
