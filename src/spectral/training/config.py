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


@dataclass
class PretrainConfig(YamlConfig):
    """Self-supervised masked-pretraining loop (Phase 3). No labels used."""

    n_pretrain: int = 20000
    pretrain_seed: int = 20000   # mixture stream disjoint from train (0) and val (10001)
    n_recon_examples: int = 4     # samples shown in the reconstruction sanity figure
    batch_size: int = 128
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.01
    mask_ratio: float = 0.5       # fraction of patches hidden per sample
    span_len: int = 4             # masked patches come in contiguous spans of this length
    seed: int = 0
    device: str = "auto"
    encoder_out: str = "experiments/pretrained_encoder.pt"  # weights for Phase-4 fine-tuning


@dataclass
class PretrainExperimentConfig(YamlConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
