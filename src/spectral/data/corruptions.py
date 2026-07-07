"""Realistic corruptions layered on top of the clean sum of components.

Each function is pure: it takes the clean signal and an RNG and returns an array, so the
generator can add them and keep an exact record (mixture = clean + baseline + noise, when
phase distortion is off - which is the default). Peak-position *jitter* is applied earlier,
at render time in the generator, so it lands inside the ground-truth component signals.
"""

from __future__ import annotations

import numpy as np

from spectral.data.config import CorruptionConfig


def make_baseline(
    grid: np.ndarray, clean: np.ndarray, cfg: CorruptionConfig, rng: np.random.Generator
) -> np.ndarray:
    """Smooth low-frequency baseline drift, scaled to the tallest peak.

    Modeled as a sum of a few long-wavelength sinusoids - the slow rolling baseline that
    real spectra pick up from the instrument.
    """
    peak_height = float(np.max(clean)) if clean.size else 1.0
    if peak_height <= 0.0:
        peak_height = 1.0
    span = float(grid[-1] - grid[0])
    baseline = np.zeros_like(grid)
    for _ in range(cfg.baseline_n_waves):
        wavelength = rng.uniform(0.5, 1.5) * span  # long compared to peak widths
        k = 2.0 * np.pi / wavelength
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = cfg.baseline_frac * peak_height / cfg.baseline_n_waves
        baseline += amp * np.sin(k * grid + phase)
    return baseline


def make_noise(clean: np.ndarray, cfg: CorruptionConfig, rng: np.random.Generator) -> np.ndarray:
    """Additive white Gaussian noise at the configured SNR.

    SNR here is the NMR convention: tallest-peak-height / noise-std. So
    noise_std = peak_height / snr.
    """
    peak_height = float(np.max(clean)) if clean.size else 1.0
    if peak_height <= 0.0:
        peak_height = 1.0
    noise_std = peak_height / cfg.snr
    return rng.normal(0.0, noise_std, size=clean.shape)


def apply_phase(signal: np.ndarray, phase_deg: float) -> np.ndarray:
    """Apply a zero-order phase error (degrees). 0 -> identity (the default).

    A real absorption spectrum acquires `cos(phi)*absorption + sin(phi)*dispersion` under a
    phase error, where the dispersion is the Hilbert transform. Off by default; when on, it
    shows up as an antisymmetric distortion in the reconstruction residual.
    """
    if phase_deg == 0.0:
        return signal.copy()
    phi = np.deg2rad(phase_deg)
    return np.cos(phi) * signal + np.sin(phi) * _dispersion(signal)


def _dispersion(x: np.ndarray) -> np.ndarray:
    """Imaginary part of the analytic signal (Hilbert transform) via FFT."""
    n = x.shape[0]
    xf = np.fft.fft(x)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(xf * h).imag
