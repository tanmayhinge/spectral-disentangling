"""Helpers to materialize labeled tensors for supervised training.

The generator is reproducible per (base_seed, index), so we pre-generate a fixed pool of
(mixture, presence) pairs once and reuse it every epoch -- faster than regenerating, and
still fully reproducible. Train and validation use different base seeds so their mixtures
never overlap.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from spectral.data.config import DataConfig
from spectral.data.generator import MixtureGenerator
from spectral.data.library import CompoundLibrary


def build_presence_tensors(
    data_cfg: DataConfig, library: CompoundLibrary, base_seed: int, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate `n` mixtures with the given base seed -> (X: (n, N), Y: (n, M))."""
    cfg = copy.deepcopy(data_cfg)
    cfg.base_seed = base_seed
    gen = MixtureGenerator(cfg, library)

    n_points = cfg.grid.n_points
    m = library.n_compounds
    x = np.zeros((n, n_points), dtype=np.float32)
    y = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        sample = gen.generate(i)
        x[i] = sample.mixture
        y[i] = sample.present
    return torch.from_numpy(x), torch.from_numpy(y)


def build_mixture_tensor(
    data_cfg: DataConfig, library: CompoundLibrary, base_seed: int, n: int
) -> torch.Tensor:
    """Generate `n` unlabeled mixtures (just the observed signal) -> X: (n, N).

    Used for self-supervised pretraining, which never touches the labels.
    """
    cfg = copy.deepcopy(data_cfg)
    cfg.base_seed = base_seed
    gen = MixtureGenerator(cfg, library)

    x = np.zeros((n, cfg.grid.n_points), dtype=np.float32)
    for i in range(n):
        x[i] = gen.generate(i).mixture
    return torch.from_numpy(x)
