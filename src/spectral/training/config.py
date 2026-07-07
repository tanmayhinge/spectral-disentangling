"""Training configuration and the top-level experiment config.

`ExperimentConfig` composes the three pieces -- data factory, model, and training loop --
so one YAML file fully specifies a run. Nested dataclasses are handled by YamlConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from spectral.config import YamlConfig
from spectral.data.config import DataConfig
from spectral.models.config import ModelConfig


@dataclass
class TrainConfig(YamlConfig):
    n_train: int = 4000
    n_val: int = 1000
    val_seed: int = 10_001    # separate mixture stream from data.base_seed -> disjoint val set
    batch_size: int = 64
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 0.01
    threshold: float = 0.5    # sigmoid decision threshold for presence
    seed: int = 0             # seeds model init + data ordering
    device: str = "auto"      # "auto" | "cpu" | "cuda" | "mps"


@dataclass
class ExperimentConfig(YamlConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
