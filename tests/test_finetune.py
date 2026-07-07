"""Tests for the Phase-4 fine-tune runner: both init paths run and return a valid score."""

import torch

from spectral.models.config import ModelConfig
from spectral.models.transformer import SpectralEncoder
from spectral.training.finetune import finetune_once


def _cfg():
    return ModelConfig(patch_size=32, d_model=32, n_heads=4, n_layers=1, dim_feedforward=64)


def _tiny_data():
    torch.manual_seed(0)
    x_train = torch.randn(8, 2048)
    y_train = (torch.rand(8, 12) > 0.5).float()
    x_val = torch.randn(8, 2048)
    y_val = (torch.rand(8, 12) > 0.5).float()
    return x_train, y_train, x_val, y_val


def _run(init, pretrained_path):
    xt, yt, xv, yv = _tiny_data()
    return finetune_once(
        _cfg(), 2048, 12, xt, yt, xv, yv,
        init=init, seed=0, max_steps=4, eval_every=2, batch_size=8,
        lr=1e-3, weight_decay=0.0, threshold=0.5,
        pretrained_encoder=pretrained_path, device=torch.device("cpu"),
    )


def test_finetune_scratch_returns_valid_f1():
    f1 = _run("scratch", "unused.pt")
    assert 0.0 <= f1 <= 1.0


def test_finetune_pretrained_loads_encoder(tmp_path):
    # Save a compatible encoder state dict, then fine-tune from it.
    enc_path = tmp_path / "enc.pt"
    torch.save(SpectralEncoder(_cfg(), 2048).state_dict(), enc_path)
    f1 = _run("pretrained", str(enc_path))
    assert 0.0 <= f1 <= 1.0


def test_finetune_rejects_bad_init():
    try:
        _run("banana", "unused.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown init")
