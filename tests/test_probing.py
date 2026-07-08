"""Tests for Phase-5 probing: feature extraction is frozen, and probes return valid metrics."""

import torch

from spectral.models.config import ModelConfig
from spectral.models.transformer import SpectralEncoder
from spectral.probing.features import extract_features
from spectral.probing.probes import (
    probe_concentration,
    probe_count,
    probe_presence,
    standardize,
)


def _cfg():
    return ModelConfig(patch_size=32, d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)


def test_extract_features_shape_and_frozen():
    enc = SpectralEncoder(_cfg(), 2048)
    x = torch.randn(10, 2048)
    feats = extract_features(enc, x, torch.device("cpu"), pool="mean")
    assert feats.shape == (10, 32)
    assert not feats.requires_grad                       # frozen: no autograd graph
    # Encoder weights must be untouched by extraction (probe never trains the encoder).
    before = enc.cls_token.clone()
    extract_features(enc, x, torch.device("cpu"))
    assert torch.equal(before, enc.cls_token)


def test_standardize_zero_mean_unit_std():
    tr = torch.randn(100, 8) * 3 + 5
    va = torch.randn(20, 8)
    tr_s, _ = standardize(tr, va)
    assert torch.allclose(tr_s.mean(0), torch.zeros(8), atol=1e-5)
    assert torch.allclose(tr_s.std(0), torch.ones(8), atol=1e-2)


def _synthetic_probe_data():
    """Features linearly predictive of the labels, so a probe should score well."""
    torch.manual_seed(0)
    d, n = 16, 400
    present = (torch.rand(n, 12) > 0.6).float()
    w = torch.randn(12, d)
    feat = present @ w + 0.05 * torch.randn(n, d)         # linear signal + small noise
    conc = present * torch.rand(n, 12)
    return feat, present, conc


def test_probes_recover_linear_signal():
    feat, present, conc = _synthetic_probe_data()
    ftr, fva = feat[:300], feat[300:]
    ptr, pva = present[:300], present[300:]
    ctr, cva = conc[:300], conc[300:]

    f1 = probe_presence(ftr, ptr, fva, pva, steps=300, lr=0.05, threshold=0.5, seed=0)
    acc = probe_count(ftr, ptr, fva, pva, steps=300, lr=0.05, seed=0)
    mae = probe_concentration(ftr, ctr, fva, cva, steps=300, lr=0.05, seed=0)

    assert f1 > 0.8          # presence is linearly decodable from these features
    assert 0.0 <= acc <= 1.0
    assert mae < 0.5         # better than predicting a constant
