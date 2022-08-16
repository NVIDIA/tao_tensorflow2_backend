# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

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
class OptimConfig:
    """Learning rate config."""

    optimizer: str = 'sgd'
    lr: float = 0.05
    decay: float = 0.0001
    epsilon: float = 0.0001
    rho: float = 0.5
    beta_1: float = 0.99
    beta_2: float = 0.99
    momentum: float = 0.99
    nesterov: bool = True

@dataclass
class LRConfig:
    """Learning rate config."""

    scheduler: str = 'cosine' # soft_anneal, step
    learning_rate: float = 0.05
    soft_start: float = 0.05
    annealing_points: List[float] = field(default_factory=lambda: [0.33, 0.66, 0.88])
    annealing_divider: float = 10
    min_lr_ratio: float = 0.00003
    gamma: float = 0.000015
    step_size: int = 10


@dataclass
class TrainConfig:
    """Train config."""
    qat: bool = True
    train_dataset_path: str = MISSING
    val_dataset_path: str = MISSING
    pretrained_model_path: str = ''
    checkpoint_freq: int = 1
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
    lr_config: LRConfig = LRConfig()
    optim_config: OptimConfig = OptimConfig()


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
    retain_head: bool = False
    dropout: float = 0.0
    resize_interpolation_method: str = 'bilinear' # or 'bicubic'
    byom_model: str = ''


@dataclass
class EvalConfig:
    """Experiment config."""

    eval_dataset_path: str = ''
    model_path: str = ''
    batch_size: int = 64
    n_workers: int = 64
    enable_center_crop: bool = True
    top_k: int = 3
    classmap: str = MISSING


@dataclass
class ExportConfig:
    """Export config."""

    model_path: str = ''
    output_path: str = ''
    data_type: str = "fp32"
    engine_file: str = ""
    max_workspace_size: int = 2  # in Gb
    cal_image_dir: str = ""
    cal_cache_file: str = ""
    cal_data_file: str = ""
    batch_size: int = 16
    batches: int = 10
    max_batch_size: int = 1
    min_batch_size: int = 1
    opt_batch_size: int = 1
    verbose: bool = False


@dataclass
class InferConfig:
    """Export config."""

    model_path: str = ''
    image_dir: str = ''

@dataclass
class PruneConfig:
    """Pruning config."""
    
    model_path: str = MISSING
    byom_model_path: str = MISSING
    normalizer: str = 'max'
    output_path: str = MISSING
    equalization_criterion: str = 'union'
    pruning_granularity: int = 8
    pruning_threshold: float = MISSING
    min_num_filters: int = 16
    excluded_layers: List[str] = field(default_factory=lambda: [])
    verbose: bool = True

@dataclass
class ExperimentConfig:
    """Experiment config."""

    train_config: TrainConfig = TrainConfig()
    model_config: ModelConfig = ModelConfig()
    eval_config: EvalConfig = EvalConfig()
    export_config: ExportConfig = ExportConfig()
    infer_config: InferConfig = InferConfig()
    prune_config: PruneConfig = PruneConfig()
    results_dir: str = MISSING
    key: str = 'nvidia_tlt'
    init_epoch: int = 1
    data_format: str = 'channels_last'
