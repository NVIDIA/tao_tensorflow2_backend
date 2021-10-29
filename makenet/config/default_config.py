# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class TrainConfig:
    """Train config."""

    train_dataset_path: str = MISSING
    val_dataset_path: str = MISSING
    pretrained_model_path: str = ''
    batch_size_per_gpu: int = 16
    n_epochs: int = 100
    n_workers: int = 10
    random_seed: int = 42
    enable_random_crop: bool = True
    enable_center_crop: bool = True
    enable_color_augmentation: bool = False
    label_smoothing: float = 0.01
    preprocess_mode: str = 'torch'
    mixup_alpha: float = 0
    disable_horizontal_flip: bool = False


@dataclass
class ModelConfig:
    """Model config."""

    arch: str = 'resnet'
    input_image_size: List[int] = field(default_factory=lambda: [3, 224, 224])
    n_layers: int = 18
    use_batch_norm: bool = True
    use_bias: bool = False
    use_pooling: bool = True
    all_projections: bool = False
    freeze_bn: bool = False
    freeze_blocks: List[int] = field(default_factory=lambda: []) # TODO
    use_imagenet_head: bool = False
    dropout: float = 0.0


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train_config: TrainConfig = TrainConfig()
    model_config: ModelConfig = ModelConfig()
    results_dir: str = MISSING
    key: str = ''
    init_epoch: int = 1
