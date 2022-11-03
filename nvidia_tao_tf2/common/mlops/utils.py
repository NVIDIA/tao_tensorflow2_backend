# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""MLOps utils."""
import logging

from nvidia_tao_tf2.common.mlops.clearml import get_clearml_task
from nvidia_tao_tf2.common.mlops.wandb import check_wandb_logged_in, initialize_wandb

logger = logging.getLogger(__name__)


def init_mlops(cfg, name):
    """Initialize mlops components."""
    wandb_logged_in = check_wandb_logged_in()
    if wandb_logged_in:
        wandb_name = cfg.train.wandb.name if cfg.train.wandb.name else f"{name}_train"
        initialize_wandb(
            project=cfg.train.wandb.project,
            entity=cfg.train.wandb.entity,
            name=wandb_name,
            wandb_logged_in=wandb_logged_in,
            config=cfg,
            results_dir=cfg.results_dir
        )
    if cfg.train.get("clearml", None):
        logger.info("Setting up communication with ClearML server.")
        get_clearml_task(
            cfg.train.clearml,
            network_name=name,
            action="train"
        )
