# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class LoaderConfig:
    shuffle_buffer: int = 10000
    cycle_length: int = 32
    block_length: int = 16
    shuffle_file: bool = True
    prefetch_size: int = 2


@dataclass
class LRConfig:
    name: str = 'cosine' # soft_anneal
    warmup_epoch: int = 5
    warmup_init: float = 0.0001
    learning_rate: float = 0.2


@dataclass
class OptConfig:
    name: str = 'sgd'
    momentum: float = 0.9


@dataclass
class TrainConfig:
    """Train config."""
    optimizer: OptConfig = OptConfig()
    lr_schedule: LRConfig = LRConfig()
    iterations_per_loop: int = 10
    num_examples_per_epoch: int = 120000
    batch_size: int = 8
    num_epochs: int = 300
    checkpoint: str = ""
    tf_random_seed: int = 42
    l1_weight_decay: float = 0.0
    l2_weight_decay: float = 0.00004
    amp: bool = False
    pruned_model_path: str = ''
    moving_average_decay: float = 0.9999
    clip_gradients_norm: float = 10.0
    skip_checkpoint_variables: str = ''
    checkpoint_period: int = 10
    image_preview: bool = True
    qat: bool = False
    results_dir: str = MISSING


@dataclass
class ModelConfig:
    """Model config."""

    name: str = 'efficientdet-d0'
    aspect_ratios: str = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    anchor_scale: int = 4
    min_level: int = 3
    max_level: int = 7
    num_scales: int = 3
    freeze_bn: bool = False
    freeze_blocks: List[int] = field(default_factory=lambda: []) # TODO


@dataclass
class DataConfig:
    """Data config."""
    train_tfrecords: List[str] = field(default_factory=lambda: []) # TODO
    train_dirs: List[str] = field(default_factory=lambda: []) # TODO
    val_tfrecords: List[str] = field(default_factory=lambda: []) # TODO
    val_dirs: List[str] = field(default_factory=lambda: []) # TODO
    val_json_file: str = ""
    testdev_dir: str = ''
    num_classes: int = 91
    max_instances_per_image: int = 200
    skip_crowd_during_training: bool = True
    use_fake_data: bool = False
    image_size: str = '512x512' # TODO
    loader: LoaderConfig = LoaderConfig()


@dataclass
class EvalConfig:
    """Eval config."""

    batch_size: int = 8
    min_score_thresh: float = 0.3
    eval_epoch_cycle: int = 10
    num_samples: int = 5000
    max_detections_per_image: int = 100
    label_map: str = ''
    iou_thresh: float = 0.5
    max_nms_inputs: int = 5000
    model_path: str = ''


@dataclass
class AugmentationConfig:
    """Augmentation config."""

    rand_hflip: bool = True
    random_crop_min_scale: float = 0.1
    random_crop_max_scale: float = 2

@dataclass
class ExportConfig:
    """Export config."""

    max_batch_size: int = 8
    model_path: str = MISSING
    output_path: str = MISSING
    engine_file: str = ""
    data_type: str = "fp32"
    max_workspace_size: int = 2 # G
    cal_image_dir: str = ""
    cal_cache_file: str = ""
    cal_batch_size: int = 16
    cal_batches: int = 10

@dataclass
class InferenceConfig:
    """Inference config."""
    
    model_path: str = MISSING
    image_dir: str = MISSING
    output_dir: str = MISSING
    dump_label: bool = False
    batch_size: int = 1

@dataclass
class PruneConfig:
    """Pruning config."""
    
    model_path: str = MISSING
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
    model: ModelConfig = ModelConfig()
    evaluate: EvalConfig = EvalConfig()
    data: DataConfig = DataConfig()
    augment: AugmentationConfig = AugmentationConfig()
    export: ExportConfig = ExportConfig()
    inference: InferenceConfig = InferenceConfig()
    prune: PruneConfig = PruneConfig()
    key: str = ''
    data_format: str = 'channels_last'
    verbose: bool = False
