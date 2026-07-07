"""Tests for the Phase-2 model and metrics.

Covers input->output shapes, the patch-count math, metric correctness on a hand-checked
example, and a tiny overfit run proving gradients actually flow and the loss goes down.
"""

import torch

from spectral.models.config import ModelConfig
from spectral.models.transformer import PresenceClassifier, SpectralEncoder, count_parameters
from spectral.training.metrics import presence_scores


def _cfg():
    return ModelConfig(patch_size=32, d_model=32, n_heads=4, n_layers=2, dim_feedforward=64)


def test_encoder_shape_and_patch_count():
    enc = SpectralEncoder(_cfg(), n_points=2048)
    assert enc.n_patches == 2048 // 32
    out = enc(torch.randn(3, 2048))
    assert out.shape == (3, enc.n_patches + 1, 32)  # +1 CLS token


def test_classifier_shape():
    model = PresenceClassifier(_cfg(), n_points=2048, n_classes=12)
    logits = model(torch.randn(5, 2048))
    assert logits.shape == (5, 12)
    assert count_parameters(model) > 0


def test_patch_divisibility_checked():
    try:
        SpectralEncoder(ModelConfig(patch_size=30), n_points=2048)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError when n_points not divisible by patch_size")


def test_presence_scores_perfect_and_known():
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    perfect = torch.where(targets == 1, 10.0, -10.0)  # confident correct logits
    s = presence_scores(perfect, targets)
    assert s.macro_f1 == 1.0 and s.exact_match == 1.0

    # One false negative on class 0: pred all-negative for row 0.
    logits = torch.tensor([[-10.0, -10.0, 10.0], [-10.0, 10.0, -10.0]])
    s2 = presence_scores(logits, targets)
    assert s2.exact_match == 0.5              # row 1 correct, row 0 wrong
    assert 0.0 < s2.macro_f1 < 1.0


def test_overfits_tiny_batch():
    """A few steps on 8 fixed samples should drive BCE loss well down."""
    torch.manual_seed(0)
    model = PresenceClassifier(_cfg(), n_points=2048, n_classes=12)
    x = torch.randn(8, 2048)
    y = (torch.rand(8, 12) > 0.5).float()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.BCEWithLogitsLoss()

    first = crit(model(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
