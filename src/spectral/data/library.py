"""The fixed compound library - the set of component "fingerprints" mixtures draw from.

Each compound is a small list of peaks. The library is generated once from a seed (with
peak positions biased toward realistic 1H regions) and frozen to a JSON file, so every
experiment sees the *same* 12 compounds. That fixed identity is what makes "which
components are present" a well-defined classification target.

`CompoundLibrary` is deliberately grid-agnostic: it stores peak parameters and renders a
compound onto whatever grid you pass in.

Extensibility: to swap in real single-compound spectra later (e.g. nmrshiftdb2), provide
an object with the same `n_compounds` / `render(index, grid, ppm_shift)` interface and the
generator won't know the difference.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from spectral.data.config import LibraryConfig
from spectral.data.lineshapes import render_multiplet
from spectral.utils import PROJECT_ROOT


@dataclass
class Peak:
    """One (possibly split) resonance."""

    center_ppm: float
    amp: float
    fwhm_ppm: float
    multiplicity: int
    j_hz: float


@dataclass
class Compound:
    """A named component: an index and its list of peaks."""

    index: int
    peaks: list[Peak]


class CompoundLibrary:
    """A frozen collection of compounds plus the render settings they were built with."""

    def __init__(self, compounds: list[Compound], lineshape: str, spectrometer_mhz: float):
        self.compounds = compounds
        self.lineshape = lineshape
        self.spectrometer_mhz = spectrometer_mhz

    @property
    def n_compounds(self) -> int:
        return len(self.compounds)

    # ---- construction -----------------------------------------------------------------
    @classmethod
    def generate(cls, cfg: LibraryConfig) -> "CompoundLibrary":
        """Build the library deterministically from `cfg.seed` (does not touch disk)."""
        rng = np.random.default_rng(cfg.seed)
        region_weights = np.array([r["weight"] for r in cfg.regions], float)
        region_weights = region_weights / region_weights.sum()
        mult_probs = np.array(cfg.multiplicity_probs, float)
        mult_probs = mult_probs / mult_probs.sum()

        compounds: list[Compound] = []
        for idx in range(cfg.n_compounds):
            n_peaks = int(rng.integers(cfg.peaks_min, cfg.peaks_max + 1))
            peaks: list[Peak] = []
            for _ in range(n_peaks):
                region = cfg.regions[rng.choice(len(cfg.regions), p=region_weights)]
                center = float(rng.uniform(region["ppm_min"], region["ppm_max"]))
                amp = float(rng.uniform(cfg.amp_min, cfg.amp_max))
                fwhm = float(rng.uniform(cfg.width_min_ppm, cfg.width_max_ppm))
                mult = int(rng.choice(cfg.multiplicities, p=mult_probs))
                j_hz = float(rng.uniform(cfg.j_min_hz, cfg.j_max_hz)) if mult > 1 else 0.0
                peaks.append(Peak(center, amp, fwhm, mult, j_hz))
            compounds.append(Compound(index=idx, peaks=peaks))

        return cls(compounds, cfg.lineshape, cfg.spectrometer_mhz)

    @classmethod
    def from_config(cls, cfg: LibraryConfig) -> "CompoundLibrary":
        """Load the frozen library from `cfg.path`, or generate+freeze it if absent.

        Freezing means the compounds stay stable even if generation code later changes.
        To regenerate, delete the JSON file.
        """
        path = _resolve(cfg.path)
        if path.exists():
            return cls.load(path)
        lib = cls.generate(cfg)
        lib.save(path)
        return lib

    # ---- persistence ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = _resolve(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lineshape": self.lineshape,
            "spectrometer_mhz": self.spectrometer_mhz,
            "compounds": [
                {"index": c.index, "peaks": [asdict(p) for p in c.peaks]}
                for c in self.compounds
            ],
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CompoundLibrary":
        with open(_resolve(path), "r") as f:
            payload = json.load(f)
        compounds = [
            Compound(index=c["index"], peaks=[Peak(**pk) for pk in c["peaks"]])
            for c in payload["compounds"]
        ]
        return cls(compounds, payload["lineshape"], payload["spectrometer_mhz"])

    # ---- rendering --------------------------------------------------------------------
    def render(self, index: int, grid: np.ndarray, ppm_shift: float = 0.0) -> np.ndarray:
        """Render compound `index` onto `grid`, optionally shifted by `ppm_shift` (jitter).

        The shift moves every peak of the compound by the same amount, modeling a small
        calibration wobble.
        """
        signal = np.zeros_like(grid)
        for peak in self.compounds[index].peaks:
            j_ppm = peak.j_hz / self.spectrometer_mhz
            signal += render_multiplet(
                grid,
                center=peak.center_ppm + ppm_shift,
                fwhm=peak.fwhm_ppm,
                amp=peak.amp,
                multiplicity=peak.multiplicity,
                j_ppm=j_ppm,
                shape=self.lineshape,
            )
        return signal


def _resolve(path: str | Path) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
