"""Model configuration.

The signal is turned into a sequence by non-overlapping patching (see transformer.py for
the rationale), so the two key size knobs are `patch_size` (how many grid points per token)
and `d_model` (token embedding width). Defaults give a ~0.5M-parameter encoder that trains
in seconds on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass

from spectral.config import YamlConfig


@dataclass
class ModelConfig(YamlConfig):
    patch_size: int = 32       # grid points per token; n_points must be divisible by this
    d_model: int = 128         # token embedding width
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    pool: str = "cls"          # "cls" (use the CLS token) or "mean" (average all tokens)
