"""Small shared helpers. Kept deliberately tiny; grows only as real needs appear."""

from __future__ import annotations

from pathlib import Path

import torch

# Repo root = two levels up from this file (src/spectral/utils.py -> repo/).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_device() -> torch.device:
    """Pick the best available compute device: CUDA, then Apple MPS, then CPU.

    Development happens on Apple Silicon (MPS); training targets a single CUDA GPU. The
    same code runs in both places - only the returned device differs.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
