"""Extract frozen features from an encoder for linear probing.

A probe asks: is some property *linearly* readable from the representation? To answer that
fairly, the encoder must be frozen -- we take its outputs as fixed features and only train a
linear layer on top. We mean-pool the patch tokens (excluding the task-specific CLS) so no
encoder is unfairly privileged by a head it happened to train.
"""

from __future__ import annotations

import torch

from spectral.models.transformer import SpectralEncoder


@torch.no_grad()
def extract_features(
    encoder: SpectralEncoder,
    x: torch.Tensor,
    device: torch.device,
    pool: str = "mean",
    batch_size: int = 256,
) -> torch.Tensor:
    """Run the frozen encoder over x (N, n_points) -> features (N, d_model) on CPU."""
    encoder.eval().to(device)
    feats = []
    for i in range(0, x.size(0), batch_size):
        tokens = encoder(x[i : i + batch_size].to(device))  # (B, n_patches+1, d)
        if pool == "cls":
            pooled = tokens[:, 0]
        elif pool == "mean":
            pooled = tokens[:, 1:].mean(dim=1)               # over patch tokens only
        else:
            raise ValueError(f"pool must be 'mean' or 'cls', got {pool!r}")
        feats.append(pooled.cpu())
    return torch.cat(feats)
