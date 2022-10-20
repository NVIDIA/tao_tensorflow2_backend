# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Default config file"""

from typing import List
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
    """Optimizer config."""

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

    scheduler: str = 'cosine'  # soft_anneal, step
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
    pretrained_model_path: str = ''
    checkpoint_freq: int = 1
    batch_size_per_gpu: int = 64
    n_epochs: int = 100
    n_workers: int = 10
    random_seed: int = 42
    label_smoothing: float = 0.01
    reg_config: RegConfig = RegConfig()
    lr_config: LRConfig = LRConfig()
    optim_config: OptimConfig = OptimConfig()


@dataclass
class AugmentConfig:
    """Augment config."""

    enable_random_crop: bool = True
    enable_center_crop: bool = True
    enable_color_augmentation: bool = False
    disable_horizontal_flip: bool = False
    mixup_alpha: float = 0


@dataclass
class DataConfig:
    """Data config."""

    train_dataset_path: str = MISSING
    val_dataset_path: str = MISSING
    preprocess_mode: str = 'caffe'
    image_mean: List[float] = field(default_factory=lambda: [103.939, 116.779, 123.68])


@dataclass
class ModelConfig:
    """Model config."""

    arch: str = 'resnet'
    input_image_size: List[int] = field(default_factory=lambda: [3, 224, 224])
    input_image_depth: int = 8
    n_layers: int = 18
    use_batch_norm: bool = True
    use_bias: bool = False
    use_pooling: bool = True
    all_projections: bool = False
    freeze_bn: bool = False
    freeze_blocks: List[int] = field(default_factory=lambda: [])
    retain_head: bool = False
    dropout: float = 0.0
    resize_interpolation_method: str = 'bilinear'  # 'bicubic'
    byom_model: str = ''


@dataclass
class EvalConfig:
    """Eval config."""

    dataset_path: str = ''
    model_path: str = ''
    batch_size: int = 64
    n_workers: int = 64
    top_k: int = 3
    classmap: str = ""


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


@dataclass
class InferConfig:
    """Inference config."""

    model_path: str = ''
    image_dir: str = ''
    classmap: str = ''


@dataclass
class PruneConfig:
    """Pruning config."""

    model_path: str = MISSING
    byom_model_path: str = MISSING
    normalizer: str = 'max'
    output_path: str = MISSING
    equalization_criterion: str = 'union'
    granularity: int = 8
    threshold: float = MISSING
    min_num_filters: int = 16
    excluded_layers: List[str] = field(default_factory=lambda: [])


@dataclass
class ExperimentConfig:
    """Experiment config."""

    train: TrainConfig = TrainConfig()
    augment: AugmentConfig = AugmentConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    evaluate: EvalConfig = EvalConfig()
    export: ExportConfig = ExportConfig()
    inference: InferConfig = InferConfig()
    prune: PruneConfig = PruneConfig()
    results_dir: str = MISSING
    key: str = 'nvidia_tlt'
    data_format: str = 'channels_first'
    verbose: bool = False
