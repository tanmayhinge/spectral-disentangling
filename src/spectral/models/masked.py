"""Masked spectrum modeling for self-supervised pretraining (Phase 3).

The idea: hide random spans of the spectrum from the model and train it to fill them back
in, using no labels at all. To reconstruct a hidden region the model must learn what
spectra look like -- where peaks sit, that they come in compound-specific groups, how wide
they are -- which is exactly the structure we hope transfers to the labeled tasks in Phase 4.

Mechanics: we patch-embed the signal (same 64 patches as Phase 2), replace the embeddings of
masked patches with a single learned `mask_token`, run the shared `SpectralEncoder`, and a
linear head predicts the raw values of every patch. The loss is computed on masked patches
only -- the model gets no credit for copying the patches it can already see.

We mask contiguous *spans* of patches rather than isolated ones: neighboring points are
highly correlated, so hiding single patches would be trivially solvable by interpolation.
Spans force the model to use longer-range structure.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from spectral.models.config import ModelConfig
from spectral.models.transformer import SpectralEncoder


def make_span_mask(n_patches: int, ratio: float, span_len: int, rng: np.random.Generator) -> np.ndarray:
    """Boolean (n_patches,) mask with ~`ratio` of patches hidden, in spans of `span_len`."""
    target = max(1, round(ratio * n_patches))
    mask = np.zeros(n_patches, dtype=bool)
    while mask.sum() < target:
        start = int(rng.integers(0, n_patches))
        mask[start : start + span_len] = True  # numpy clips the end index
    return mask


def make_batch_mask(batch: int, n_patches: int, ratio: float, span_len: int,
                    rng: np.random.Generator) -> torch.Tensor:
    """Per-sample span masks stacked into a (batch, n_patches) bool tensor."""
    masks = np.stack([make_span_mask(n_patches, ratio, span_len, rng) for _ in range(batch)])
    return torch.from_numpy(masks)


class MaskedSpectralModel(nn.Module):
    """SpectralEncoder + a learned mask token + a linear patch-reconstruction head."""

    def __init__(self, cfg: ModelConfig, n_points: int):
        super().__init__()
        self.encoder = SpectralEncoder(cfg, n_points)
        self.patch_size = cfg.patch_size
        self.n_patches = self.encoder.n_patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.recon_head = nn.Linear(cfg.d_model, cfg.patch_size)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, n_points), patch_mask: (B, n_patches) bool -> recon (B, n_patches, patch_size).

        Masked patch embeddings are replaced by the mask token before encoding.
        """
        pe = self.encoder.embed_patches(x)                       # (B, P, d)
        mask_tok = self.mask_token.expand(pe.size(0), pe.size(1), -1)
        pe = torch.where(patch_mask.unsqueeze(-1), mask_tok, pe)  # hide masked patches
        tokens = self.encoder.encode_tokens(pe)                  # (B, P+1, d)
        return self.recon_head(tokens[:, 1:])                    # drop CLS -> per-patch values

    @staticmethod
    def to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """(B, n_points) -> (B, n_patches, patch_size), the reconstruction target."""
        return x.unfold(dimension=1, size=patch_size, step=patch_size)

    def masked_mse(self, x: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """Convenience: reconstruction MSE over masked patches only."""
        recon = self(x, patch_mask)
        target = self.to_patches(x, self.patch_size)
        per_patch_mse = ((recon - target) ** 2).mean(dim=-1)  # (B, P)
        return per_patch_mse[patch_mask].mean()
