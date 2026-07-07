"""The mixture generator.

Given the fixed compound library, it produces (mixture, ground-truth-labels) pairs. Each
sample is fully reproducible from ``(base_seed, index)``, so we never store a dataset - we
regenerate any sample on demand.

The construction guarantees the key invariant used at Checkpoint 1:

    mixture = clean_mixture + baseline + noise         (with phase distortion off)
    clean_mixture = sum over the per-component signals

so "sum of components (+ corruptions) = mixture" holds to floating-point tolerance, and
the residual `mixture - clean_mixture` is exactly the corruption we added.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spectral.data import corruptions
from spectral.data.config import DataConfig
from spectral.data.library import CompoundLibrary


@dataclass
class Mixture:
    """One generated sample: the observed signal plus complete ground truth.

    Shapes: grid/mixture/clean_mixture/baseline/noise are (N,); components is (M, N);
    present/concentrations are (M,). Rows of `components` are zero for absent compounds.
    """

    grid: np.ndarray
    mixture: np.ndarray
    clean_mixture: np.ndarray
    components: np.ndarray
    present: np.ndarray          # multi-hot, {0,1}    -> classification target
    concentrations: np.ndarray   # weights, 0 if absent -> regression target
    baseline: np.ndarray
    noise: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


class MixtureGenerator:
    """Turns the compound library into reproducible (mixture, labels) samples."""

    def __init__(self, cfg: DataConfig, library: CompoundLibrary):
        self.cfg = cfg
        self.library = library
        g = cfg.grid
        self.grid = np.linspace(g.ppm_min, g.ppm_max, g.n_points).astype(np.float64)

    def _rng(self, index: int) -> np.random.Generator:
        """Independent, reproducible RNG stream for a given sample index."""
        return np.random.default_rng([self.cfg.base_seed, index])

    def generate(self, index: int) -> Mixture:
        rng = self._rng(index)
        m = self.library.n_compounds
        mix_cfg = self.cfg.mixture
        corr = self.cfg.corruptions

        # 1. Choose K distinct compounds and positive concentrations.
        k = int(rng.integers(mix_cfg.k_min, mix_cfg.k_max + 1))
        chosen = rng.choice(m, size=k, replace=False)

        components = np.zeros((m, self.grid.shape[0]), dtype=np.float64)
        concentrations = np.zeros(m, dtype=np.float64)
        for idx in chosen:
            conc = float(rng.uniform(mix_cfg.conc_min, mix_cfg.conc_max))
            # Peak-position jitter: a small shared shift, baked into the ground-truth
            # component so the reconstruction invariant stays exact.
            shift = float(rng.normal(0.0, corr.jitter_ppm)) if corr.jitter_ppm > 0 else 0.0
            components[idx] = conc * self.library.render(int(idx), self.grid, ppm_shift=shift)
            concentrations[idx] = conc

        present = (concentrations > 0).astype(np.float64)
        clean_mixture = components.sum(axis=0)

        # 2. Layer on corruptions.
        baseline = corruptions.make_baseline(self.grid, clean_mixture, corr, rng)
        noise = corruptions.make_noise(clean_mixture, corr, rng)
        phased = corruptions.apply_phase(clean_mixture, corr.phase_deg)
        mixture = phased + baseline + noise

        meta = {
            "index": index,
            "base_seed": self.cfg.base_seed,
            "k": k,
            "chosen": sorted(int(i) for i in chosen),
            "snr": corr.snr,
            "phase_deg": corr.phase_deg,
        }
        return Mixture(
            grid=self.grid,
            mixture=mixture,
            clean_mixture=clean_mixture,
            components=components,
            present=present,
            concentrations=concentrations,
            baseline=baseline,
            noise=noise,
            meta=meta,
        )
