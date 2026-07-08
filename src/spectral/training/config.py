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


@dataclass
class FinetuneConfig(YamlConfig):
    """Phase 4: the label-efficiency sweep comparing pretrained vs from-scratch init.

    Each run fine-tunes the full model on `n` labeled examples for a fixed optimization
    budget (`max_steps`), tracking the best validation macro-F1. Scratch and pretrained
    runs share the same seed, so head init and batch order are identical -- the only
    difference is whether the encoder starts from `pretrained_encoder`.
    """

    label_sizes: list = field(default_factory=lambda: [10, 40, 160, 640, 2560])
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    max_steps: int = 600
    eval_every: int = 100
    batch_size: int = 64            # actual batch is min(batch_size, n)
    lr: float = 5e-4
    weight_decay: float = 0.01
    threshold: float = 0.5

    labeled_seed_base: int = 1000   # labeled stream per seed = base + seed (disjoint from val/pretrain)
    val_seed: int = 10001
    n_val: int = 1000

    pretrained_encoder: str = "experiments/pretrained_encoder.pt"
    device: str = "auto"


@dataclass
class FinetuneExperimentConfig(YamlConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)


@dataclass
class ProbeConfig(YamlConfig):
    """Phase 5: linear probes on FROZEN features.

    Encoders are frozen; only a linear layer is trained on top. Compares random (floor),
    pretrained (the SSL question), and supervised (trained-on-labels reference) encoders on
    three targets: presence, component count K, and concentrations. Features are mean-pooled
    over patch tokens (not the task-specific CLS) so no encoder is privileged.
    """

    probe_label_sizes: list = field(default_factory=lambda: [10, 40, 160, 640, 2560])
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    probe_val_n: int = 1000
    pool: str = "mean"           # "mean" over patch tokens, or "cls"
    threshold: float = 0.5

    probe_steps: int = 300       # linear-probe optimization steps (full-batch)
    probe_lr: float = 0.01

    include_supervised: bool = True
    supervised_train_n: int = 4000
    supervised_steps: int = 1200
    supervised_lr: float = 3e-4

    pretrained_encoder: str = "experiments/pretrained_encoder.pt"
    labeled_seed_base: int = 3000   # probe-train streams (base + seed)
    supervised_seed: int = 5000     # supervised-encoder training stream
    val_seed: int = 10001
    device: str = "auto"


@dataclass
class ProbeExperimentConfig(YamlConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
