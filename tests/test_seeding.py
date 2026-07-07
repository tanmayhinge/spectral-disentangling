"""Seeding must be deterministic: same seed -> same numbers; different seed -> different.

This is the Phase 0 guarantee the whole project leans on for reproducibility.
"""

import numpy as np
import torch

from spectral.config import RunConfig
from spectral.seeding import seed_everything


def _draw():
    """Draw one number from each RNG we seed."""
    import random

    return (random.random(), float(np.random.rand()), torch.rand(1).item())


def test_same_seed_reproduces():
    seed_everything(1234)
    first = _draw()
    seed_everything(1234)
    second = _draw()
    assert first == second


def test_different_seed_differs():
    seed_everything(1234)
    a = _draw()
    seed_everything(4321)
    b = _draw()
    assert a != b


def test_config_roundtrip(tmp_path):
    """A YAML file loads into RunConfig with the right values."""
    p = tmp_path / "cfg.yaml"
    p.write_text("seed: 7\ndeterministic: false\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.seed == 7
    assert cfg.deterministic is False


def test_config_rejects_unknown_key(tmp_path):
    """Typos in a config file should fail loudly, not be ignored."""
    p = tmp_path / "bad.yaml"
    p.write_text("seed: 7\ntypo_key: 3\n")
    try:
        RunConfig.from_yaml(p)
    except ValueError as e:
        assert "typo_key" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown key")
