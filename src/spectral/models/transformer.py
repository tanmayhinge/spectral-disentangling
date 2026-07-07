"""A small transformer that reads a 1D spectrum as a sequence of patches.

How the signal becomes model input (the Phase-2 design decision):

We split the 2048-point spectrum into non-overlapping patches of `patch_size` points and
linearly embed each patch into a `d_model` vector -- the 1D analogue of a Vision
Transformer. Why this over the alternatives:

- Raw grid (one token per point): a length-2048 sequence makes attention (O(n^2)) needlessly
  expensive, and most points are empty baseline, so it wastes capacity.
- Peak-list input (positions/heights): would require a separate peak-picking step, throwing
  away the raw signal and the very corruptions we want the model to cope with. It also
  wouldn't transfer to Phase-3 masked pretraining, which reconstructs the raw signal.
- Patched tokens: short sequence (2048/32 = 64 tokens), each token sees a local chunk of the
  spectrum, and the exact same patch layout is what we later mask in Phase 3. This is the
  choice.

The encoder is intentionally generic: it returns per-token embeddings so Phase 3/4 can bolt
different heads (reconstruction, regression) onto the same backbone.
"""

from __future__ import annotations

import torch
from torch import nn

from spectral.models.config import ModelConfig


class SpectralEncoder(nn.Module):
    """Patch-embed a 1D spectrum, add a CLS token + positions, run a transformer encoder."""

    def __init__(self, cfg: ModelConfig, n_points: int):
        super().__init__()
        if n_points % cfg.patch_size != 0:
            raise ValueError(
                f"n_points ({n_points}) must be divisible by patch_size ({cfg.patch_size})"
            )
        self.cfg = cfg
        self.n_patches = n_points // cfg.patch_size

        # A strided conv is a clean way to do non-overlapping patch embedding: each output
        # position looks at one patch of `patch_size` points and produces a d_model vector.
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=cfg.d_model,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, cfg.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: more stable for small-scale training from scratch
        )
        # enable_nested_tensor=False: we use pre-norm and fixed-length sequences, so the
        # nested-tensor fast path doesn't apply (and PyTorch warns otherwise).
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_points) -> token embeddings (B, n_patches + 1, d_model).

        Token 0 is the CLS token; tokens 1.. are the patches in order.
        """
        x = x.unsqueeze(1)                       # (B, 1, n_points)
        x = self.patch_embed(x).transpose(1, 2)  # (B, n_patches, d_model)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.encoder(x)


class PresenceClassifier(nn.Module):
    """Encoder + a linear head producing one logit per compound (multi-label presence)."""

    def __init__(self, cfg: ModelConfig, n_points: int, n_classes: int):
        super().__init__()
        self.cfg = cfg
        self.encoder = SpectralEncoder(cfg, n_points)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_points) -> logits (B, n_classes). Apply sigmoid for probabilities."""
        tokens = self.encoder(x)
        pooled = tokens[:, 0] if self.cfg.pool == "cls" else tokens.mean(dim=1)
        return self.head(self.norm(pooled))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
