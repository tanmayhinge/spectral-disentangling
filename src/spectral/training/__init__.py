"""Training: configs, metrics, and data helpers for the supervised baseline."""

from spectral.training.config import ExperimentConfig, TrainConfig
from spectral.training.data import build_presence_tensors
from spectral.training.metrics import PresenceScores, presence_scores

__all__ = [
    "ExperimentConfig",
    "TrainConfig",
    "build_presence_tensors",
    "PresenceScores",
    "presence_scores",
]
