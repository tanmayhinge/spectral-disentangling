"""Training: configs, metrics, and data helpers for the supervised baseline."""

from spectral.training.config import (
    ExperimentConfig,
    PretrainConfig,
    PretrainExperimentConfig,
    TrainConfig,
)
from spectral.training.data import build_mixture_tensor, build_presence_tensors
from spectral.training.metrics import PresenceScores, presence_scores

__all__ = [
    "ExperimentConfig",
    "TrainConfig",
    "PretrainConfig",
    "PretrainExperimentConfig",
    "build_presence_tensors",
    "build_mixture_tensor",
    "PresenceScores",
    "presence_scores",
]
