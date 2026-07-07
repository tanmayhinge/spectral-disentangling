"""Peak lineshapes and multiplet rendering.

A single NMR peak is a bump on the chemical-shift axis. Two shapes are supported:
- Lorentzian (physically correct for NMR relaxation-limited lines) - the default,
- Gaussian (useful for comparison / broadening effects).

Both are parameterized by (center, FWHM, amplitude) so `amplitude` is the peak *height*
and `FWHM` (full width at half maximum, in ppm) is directly readable off a plot.

J-coupling splits one peak into a *multiplet*: several equally spaced lines whose relative
heights follow Pascal's triangle (a doublet is 1:1, a triplet 1:2:1, a quartet 1:3:3:1).
The split lines share the parent peak's total area, so the integral is preserved.
"""

from __future__ import annotations

import math

import numpy as np

# FWHM -> Gaussian sigma:  FWHM = 2*sqrt(2*ln2) * sigma
_GAUSS_FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def lorentzian(grid: np.ndarray, center: float, fwhm: float, amp: float) -> np.ndarray:
    """Lorentzian with peak height `amp` at `center` and full width `fwhm` (ppm)."""
    gamma = fwhm / 2.0  # half width at half maximum
    return amp * gamma**2 / ((grid - center) ** 2 + gamma**2)


def gaussian(grid: np.ndarray, center: float, fwhm: float, amp: float) -> np.ndarray:
    """Gaussian with peak height `amp` at `center` and full width `fwhm` (ppm)."""
    sigma = fwhm * _GAUSS_FWHM_TO_SIGMA
    return amp * np.exp(-((grid - center) ** 2) / (2.0 * sigma**2))


_SHAPES = {"lorentzian": lorentzian, "gaussian": gaussian}


def _pascal_weights(multiplicity: int) -> np.ndarray:
    """Normalized binomial intensities for an m-line multiplet (sum to 1)."""
    row = np.array([math.comb(multiplicity - 1, i) for i in range(multiplicity)], float)
    return row / row.sum()


def render_multiplet(
    grid: np.ndarray,
    center: float,
    fwhm: float,
    amp: float,
    multiplicity: int = 1,
    j_ppm: float = 0.0,
    shape: str = "lorentzian",
) -> np.ndarray:
    """Render a (possibly split) peak onto the grid.

    The multiplet is centered on `center`; its `multiplicity` lines are spaced `j_ppm`
    apart with Pascal-triangle relative heights, together carrying the parent amplitude.
    """
    if shape not in _SHAPES:
        raise ValueError(f"unknown lineshape {shape!r}; choose from {sorted(_SHAPES)}")
    shape_fn = _SHAPES[shape]

    if multiplicity <= 1 or j_ppm <= 0.0:
        return shape_fn(grid, center, fwhm, amp)

    weights = _pascal_weights(multiplicity)
    signal = np.zeros_like(grid)
    for i, w in enumerate(weights):
        offset = (i - (multiplicity - 1) / 2.0) * j_ppm  # symmetric about center
        signal += shape_fn(grid, center + offset, fwhm, amp * w)
    return signal
