"""Tests for Phase-3 masked pretraining: masking, shapes, loss behavior, weight transfer."""

import numpy as np
import torch

from spectral.models.config import ModelConfig
from spectral.models.masked import MaskedSpectralModel, make_batch_mask, make_span_mask
from spectral.models.transformer import PresenceClassifier


def _cfg():
    return ModelConfig(patch_size=32, d_model=32, n_heads=4, n_layers=2, dim_feedforward=64)


def test_span_mask_ratio_and_contiguity():
    rng = np.random.default_rng(0)
    mask = make_span_mask(n_patches=64, ratio=0.5, span_len=4, rng=rng)
    assert mask.dtype == bool and mask.shape == (64,)
    assert 0.5 * 64 <= mask.sum() <= 0.5 * 64 + 4      # meets target, small overshoot
    assert not mask.all()                               # never masks everything


def test_batch_mask_shape():
    rng = np.random.default_rng(0)
    m = make_batch_mask(8, 64, 0.5, 4, rng)
    assert m.shape == (8, 64) and m.dtype == torch.bool


def test_reconstruction_shape():
    model = MaskedSpectralModel(_cfg(), n_points=2048)
    x = torch.randn(3, 2048)
    mask = make_batch_mask(3, model.n_patches, 0.5, 4, np.random.default_rng(0))
    recon = model(x, mask)
    assert recon.shape == (3, model.n_patches, 32)


def test_to_patches_roundtrip():
    x = torch.arange(2048).float().unsqueeze(0)
    patches = MaskedSpectralModel.to_patches(x, 32)
    assert patches.shape == (1, 64, 32)
    assert torch.equal(patches.reshape(1, -1), x)      # patches concatenate back to signal


def test_masked_mse_decreases_on_overfit():
    """Pretraining a few steps on a fixed batch should lower the masked MSE."""
    torch.manual_seed(0)
    model = MaskedSpectralModel(_cfg(), n_points=2048)
    x = torch.randn(6, 2048)
    mask = make_batch_mask(6, model.n_patches, 0.5, 4, np.random.default_rng(0))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = model.masked_mse(x, mask).item()
    for _ in range(50):
        opt.zero_grad()
        loss = model.masked_mse(x, mask)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_encoder_weights_load_into_classifier():
    """The pretrained encoder must drop cleanly into the Phase-2 classifier (transfer path)."""
    cfg = _cfg()
    pre = MaskedSpectralModel(cfg, n_points=2048)
    clf = PresenceClassifier(cfg, n_points=2048, n_classes=12)
    missing, unexpected = clf.encoder.load_state_dict(pre.encoder.state_dict(), strict=True)
    assert not missing and not unexpected
