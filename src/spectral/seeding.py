"""Reproducible seeding.

One function, called at the top of every entry point, so that a given seed produces the
same numbers everywhere (Python, NumPy, PyTorch - CPU and CUDA). Reproducibility from day
one is a Working-Agreement requirement.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed all RNGs we use and (optionally) force deterministic cuDNN.

    Args:
        seed: the integer seed to apply everywhere.
        deterministic: if True, make cuDNN deterministic. This can slow training a little
            but removes run-to-run nondeterminism from GPU convolutions. Keep it on until
            we have a reason to trade reproducibility for speed.

    Returns:
        The seed, so callers can log exactly what was used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op if CUDA is unavailable

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed
