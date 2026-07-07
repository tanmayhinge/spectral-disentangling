"""The data factory: synthetic mixed spectra with exact ground truth."""

from spectral.data.config import DataConfig
from spectral.data.dataset import MixtureDataset, dump_npz
from spectral.data.generator import Mixture, MixtureGenerator
from spectral.data.library import CompoundLibrary

__all__ = [
    "DataConfig",
    "MixtureDataset",
    "dump_npz",
    "Mixture",
    "MixtureGenerator",
    "CompoundLibrary",
]
