# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
