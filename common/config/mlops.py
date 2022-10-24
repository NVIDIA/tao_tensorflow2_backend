# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Default config file"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class WandBConfig:
    """Configuration element wandb client."""

    project: str = "TAO Toolkit"
    entity: Optional[str] = None
    tags: List[str] = field(default_factory=lambda: [])
    reinit: bool = False
    sync_tensorboard: bool = True
    save_code: bool = False
    name: str = None
