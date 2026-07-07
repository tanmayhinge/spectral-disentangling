"""Tests for the data factory - the foundation everything downstream is graded against.

The central guarantee: the mixture is exactly the sum of the ground-truth components plus
the recorded corruptions. If that ever breaks, every later metric is meaningless.
"""

import numpy as np
import pytest

from spectral.data.config import DataConfig
from spectral.data.generator import MixtureGenerator
from spectral.data.library import CompoundLibrary
from spectral.data.lineshapes import gaussian, lorentzian, render_multiplet

CONFIG = "configs/data/default.yaml"


@pytest.fixture(scope="module")
def gen():
    cfg = DataConfig.from_yaml(CONFIG)
    library = CompoundLibrary.generate(cfg.library)  # in-memory, no disk
    return MixtureGenerator(cfg, library)


# ---- lineshapes -----------------------------------------------------------------------
def test_lorentzian_height_and_fwhm():
    grid = np.linspace(0, 10, 4001)
    y = lorentzian(grid, center=5.0, fwhm=0.2, amp=1.0)
    assert y.max() == pytest.approx(1.0, abs=1e-3)          # height == amp
    half = lorentzian(np.array([5.0 + 0.1]), 5.0, 0.2, 1.0)  # at center + FWHM/2
    assert half[0] == pytest.approx(0.5, abs=1e-6)          # half maximum


def test_gaussian_height():
    grid = np.linspace(0, 10, 4001)
    y = gaussian(grid, center=5.0, fwhm=0.2, amp=0.7)
    assert y.max() == pytest.approx(0.7, abs=1e-3)


def test_multiplet_preserves_area():
    grid = np.linspace(0, 10, 20001)
    singlet = render_multiplet(grid, 5.0, 0.03, 1.0, multiplicity=1)
    triplet = render_multiplet(grid, 5.0, 0.03, 1.0, multiplicity=3, j_ppm=0.05)
    # Splitting redistributes intensity but conserves the integral.
    assert np.trapezoid(triplet, grid) == pytest.approx(np.trapezoid(singlet, grid), rel=1e-3)


# ---- generator invariants -------------------------------------------------------------
def test_shapes(gen):
    m = gen.generate(0)
    n = gen.cfg.grid.n_points
    mc = gen.library.n_compounds
    assert m.mixture.shape == (n,)
    assert m.components.shape == (mc, n)
    assert m.present.shape == (mc,)


def test_reconstruction_invariant(gen):
    """mixture == sum(components) + baseline + noise, and clean == sum(components)."""
    for i in range(20):
        m = gen.generate(i)
        assert np.allclose(m.clean_mixture, m.components.sum(axis=0))
        assert np.allclose(m.mixture, m.clean_mixture + m.baseline + m.noise)


def test_labels_consistent(gen):
    """present <-> concentration > 0 <-> component row is nonzero; K present matches meta."""
    for i in range(20):
        m = gen.generate(i)
        assert np.array_equal(m.present, (m.concentrations > 0).astype(float))
        assert int(m.present.sum()) == m.meta["k"]
        for j in range(gen.library.n_compounds):
            if m.present[j]:
                assert np.any(m.components[j] != 0)
            else:
                assert np.all(m.components[j] == 0)


def test_k_in_configured_range(gen):
    lo, hi = gen.cfg.mixture.k_min, gen.cfg.mixture.k_max
    for i in range(50):
        assert lo <= gen.generate(i).meta["k"] <= hi


def test_determinism(gen):
    a, b = gen.generate(7), gen.generate(7)
    assert np.array_equal(a.mixture, b.mixture)
    assert np.array_equal(a.concentrations, b.concentrations)


def test_different_indices_differ(gen):
    assert not np.array_equal(gen.generate(1).mixture, gen.generate(2).mixture)


def test_library_freeze_roundtrip(tmp_path, gen):
    """A saved-then-loaded library renders identically."""
    p = tmp_path / "lib.json"
    gen.library.save(p)
    reloaded = CompoundLibrary.load(p)
    assert np.allclose(gen.library.render(0, gen.grid), reloaded.render(0, gen.grid))
