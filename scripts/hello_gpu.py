"""Checkpoint 0 sanity script.

Confirms the environment is wired up: PyTorch imports, we can pick a compute device, and
seeding + config loading work. Prints a tiny report and does a trivial tensor op on the
chosen device. No ML yet.

Run:  python scripts/hello_gpu.py [--config configs/base.yaml]
"""

from __future__ import annotations

import argparse
import platform

import torch

from spectral.config import RunConfig
from spectral.seeding import seed_everything
from spectral.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Environment sanity check (Checkpoint 0).")
    parser.add_argument("--config", default="configs/base.yaml", help="path to run config")
    args = parser.parse_args()

    cfg = RunConfig.from_yaml(args.config)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    device = get_device()

    print("=== hello_gpu ===")
    print(f"python        : {platform.python_version()} ({platform.machine()})")
    print(f"torch         : {torch.__version__}")
    print(f"config        : {args.config} -> {cfg.to_dict()}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"mps available : {torch.backends.mps.is_available()}")
    print(f"device chosen : {device}")

    # Trivial op on the chosen device proves tensors actually run there.
    x = torch.randn(1000, 1000, device=device)
    y = (x @ x).sum().item()
    print(f"sanity matmul : sum={y:.3f} on {device}")
    print("environment OK")


if __name__ == "__main__":
    main()
