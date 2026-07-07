"""PyTorch Dataset wrapper around the generator, plus a helper to freeze a fixed set.

Training uses `MixtureDataset` (on-the-fly, reproducible). For stable evaluation and
figures we dump a fixed set of samples to a single `.npz` with `dump_npz`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from spectral.data.config import DataConfig
from spectral.data.generator import MixtureGenerator
from spectral.data.library import CompoundLibrary


class MixtureDataset(Dataset):
    """Virtual dataset of `cfg.n_samples` mixtures, generated on demand.

    __getitem__ returns the tensors a model needs for the Phase-2+ tasks. The heavy
    debugging arrays (baseline/noise) stay out of the training path; use the generator
    directly if you need them.
    """

    def __init__(self, cfg: DataConfig, library: CompoundLibrary | None = None):
        self.cfg = cfg
        self.library = library or CompoundLibrary.from_config(cfg.library)
        self.generator = MixtureGenerator(cfg, self.library)

    def __len__(self) -> int:
        return self.cfg.n_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        m = self.generator.generate(index)
        to_t = lambda a: torch.from_numpy(a).float()  # noqa: E731
        return {
            "mixture": to_t(m.mixture),               # (N,)
            "components": to_t(m.components),          # (M, N)
            "present": to_t(m.present),               # (M,)
            "concentrations": to_t(m.concentrations),  # (M,)
        }


def dump_npz(cfg: DataConfig, path: str | Path, n_samples: int, library=None) -> Path:
    """Generate `n_samples` fixed mixtures and save them to a single .npz file."""
    library = library or CompoundLibrary.from_config(cfg.library)
    gen = MixtureGenerator(cfg, library)
    samples = [gen.generate(i) for i in range(n_samples)]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        grid=gen.grid,
        mixture=np.stack([s.mixture for s in samples]),
        components=np.stack([s.components for s in samples]),
        present=np.stack([s.present for s in samples]),
        concentrations=np.stack([s.concentrations for s in samples]),
    )
    return path
