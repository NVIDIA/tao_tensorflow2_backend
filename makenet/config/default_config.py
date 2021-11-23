# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class RegConfig:
    """Regularizer config."""

    type: str = 'L2'
    scope: List[str] = field(default_factory=lambda: ['conv2d', 'dense'])
    weight_decay: float = 0.000015


@dataclass
class TrainConfig:
    """Train config."""
    qat: bool = True
    train_dataset_path: str = MISSING
    val_dataset_path: str = MISSING
    pretrained_model_path: str = ''
    batch_size_per_gpu: int = 64
    n_epochs: int = 100
    n_workers: int = 10
    random_seed: int = 42
    enable_random_crop: bool = True
    enable_center_crop: bool = True
    enable_color_augmentation: bool = False
    label_smoothing: float = 0.01
    preprocess_mode: str = 'caffe'
    mixup_alpha: float = 0
    disable_horizontal_flip: bool = False
    image_mean: List[float] = field(default_factory=lambda: [103.939, 116.779, 123.68])
    reg_config: RegConfig = RegConfig()


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
    resize_interpolation_method: str = 'bilinear' # or 'bicubic'


@dataclass
class EvalConfig:
    """Experiment config."""

    eval_dataset_path: str = ''
    model_path: str = ''
    batch_size: int = 64
    n_workers: int = 64
    enable_center_crop: bool = True
    top_k: int = 3


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train_config: TrainConfig = TrainConfig()
    model_config: ModelConfig = ModelConfig()
    eval_config: EvalConfig = EvalConfig()
    results_dir: str = MISSING
    key: str = ''
    init_epoch: int = 1
