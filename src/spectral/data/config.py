"""Configuration for the data factory.

Every knob the generator uses lives here as a typed dataclass field with a default, and
the actual values come from `configs/data/default.yaml`. No magic numbers in code.

The structure mirrors the YAML: a top-level `DataConfig` with nested blocks for the grid,
the compound library, mixture assembly, corruptions, and the difficulty mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectral.config import YamlConfig


@dataclass
class GridConfig(YamlConfig):
    """The shared x-axis (chemical-shift axis) every spectrum is sampled on."""

    ppm_min: float = 0.0
    ppm_max: float = 10.0
    n_points: int = 2048  # power of two -> clean patching for the transformer later


@dataclass
class LibraryConfig(YamlConfig):
    """The fixed, region-aware synthetic compound library.

    Generated once from `seed` and frozen to disk at `path`, so the same 12 compounds are
    used across every experiment regardless of the per-run seed. `regions` are (name,
    ppm_min, ppm_max, weight) bands so peaks land in realistic 1H-NMR ranges.
    """

    n_compounds: int = 12
    seed: int = 1234
    path: str = "assets/library.json"

    peaks_min: int = 2
    peaks_max: int = 5

    lineshape: str = "lorentzian"  # "lorentzian" | "gaussian"
    width_min_ppm: float = 0.02
    width_max_ppm: float = 0.045
    amp_min: float = 0.5
    amp_max: float = 1.0

    spectrometer_mhz: float = 300.0  # converts J-couplings (Hz) to ppm splittings
    j_min_hz: float = 6.0
    j_max_hz: float = 14.0

    # Peak-position priors: realistic 1H regions. Weights need not sum to 1 (normalized).
    regions: list = field(
        default_factory=lambda: [
            {"name": "aromatic", "ppm_min": 6.5, "ppm_max": 8.5, "weight": 0.35},
            {"name": "mid", "ppm_min": 3.0, "ppm_max": 4.8, "weight": 0.20},
            {"name": "aliphatic", "ppm_min": 0.5, "ppm_max": 3.0, "weight": 0.45},
        ]
    )
    # Multiplicity (number of lines) and its sampling probability. 1=singlet, 2=doublet...
    multiplicities: list = field(default_factory=lambda: [1, 2, 3, 4])
    multiplicity_probs: list = field(default_factory=lambda: [0.45, 0.30, 0.15, 0.10])


@dataclass
class MixtureConfig(YamlConfig):
    """How many compounds go into a mixture and at what concentrations."""

    k_min: int = 2
    k_max: int = 5
    conc_min: float = 0.3
    conc_max: float = 1.0


@dataclass
class CorruptionConfig(YamlConfig):
    """Realistic distortions applied on top of the clean sum of components.

    - snr: peak-height-to-noise-std ratio (NMR convention). Higher = cleaner.
    - baseline_frac: baseline drift amplitude as a fraction of the tallest peak.
    - jitter_ppm: std of a small random per-compound shift (calibration wobble).
    - phase_deg: zero-order phase error in degrees (0 = no phase distortion).
    """

    snr: float = 30.0
    baseline_frac: float = 0.05
    baseline_n_waves: int = 3
    jitter_ppm: float = 0.004
    phase_deg: float = 0.0


@dataclass
class DifficultyConfig(YamlConfig):
    """Endpoints for the single difficulty knob (see DataConfig.at_difficulty).

    difficulty=0 -> easy end, difficulty=1 -> hard end; values interpolate linearly.
    """

    snr_easy: float = 40.0
    snr_hard: float = 6.0
    kmax_easy: int = 2
    kmax_hard: int = 5
    jitter_easy_ppm: float = 0.002
    jitter_hard_ppm: float = 0.010


@dataclass
class DataConfig(YamlConfig):
    """Top-level data-factory config."""

    base_seed: int = 0        # drives per-sample mixture sampling
    n_samples: int = 10000    # virtual dataset length (samples are generated on the fly)
    grid: GridConfig = field(default_factory=GridConfig)
    library: LibraryConfig = field(default_factory=LibraryConfig)
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    corruptions: CorruptionConfig = field(default_factory=CorruptionConfig)
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)

    def at_difficulty(self, d: float) -> "DataConfig":
        """Return a copy of this config adjusted to difficulty ``d`` in [0, 1].

        Interpolates SNR, max component count, and jitter between the easy and hard
        endpoints. Used by the Phase-6 robustness sweep; individual params stay overridable.
        """
        import copy

        if not 0.0 <= d <= 1.0:
            raise ValueError(f"difficulty must be in [0, 1], got {d}")

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * d

        out = copy.deepcopy(self)
        dc = out.difficulty
        out.corruptions.snr = lerp(dc.snr_easy, dc.snr_hard)
        out.corruptions.jitter_ppm = lerp(dc.jitter_easy_ppm, dc.jitter_hard_ppm)
        out.mixture.k_max = max(out.mixture.k_min, round(lerp(dc.kmax_easy, dc.kmax_hard)))
        return out
