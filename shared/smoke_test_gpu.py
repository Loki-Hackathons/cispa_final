"""GPU smoke test for SLURM templates."""

import sys

import torch


def main() -> None:
    print(f"Python: {sys.version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPU count: {n}")
        for i in range(n):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        x = torch.randn(1000, 1000, device="cuda")
        y = x @ x.T
        print(f"Matmul OK — result shape: {y.shape}")
    else:
        print("No GPU — check SLURM allocation")
        sys.exit(1)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
