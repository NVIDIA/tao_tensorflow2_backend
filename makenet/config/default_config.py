# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""Default config file"""

from typing import Optional, List, Dict
from dataclasses import dataclass, field
from omegaconf import MISSING


@dataclass
class TrainConfig:
    """Train config."""

    train_dataset_path: str = ''
    val_dataset_path: str = ''
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
class ExperimentConfig:
    """Experiment config."""

    train_config: TrainConfig = TrainConfig()
    results_dir: str = ''
    key: str = ''
    init_epoch: int = 1
