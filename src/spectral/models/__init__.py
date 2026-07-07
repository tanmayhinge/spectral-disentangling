"""Models: a small patch-based transformer encoder and task heads."""

from spectral.models.config import ModelConfig
from spectral.models.masked import MaskedSpectralModel, make_batch_mask, make_span_mask
from spectral.models.transformer import (
    PresenceClassifier,
    SpectralEncoder,
    count_parameters,
)

__all__ = [
    "ModelConfig",
    "PresenceClassifier",
    "SpectralEncoder",
    "MaskedSpectralModel",
    "make_span_mask",
    "make_batch_mask",
    "count_parameters",
]
