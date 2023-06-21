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

"""TAO classification model pruner."""

import logging

from abc import ABC, abstractmethod
from nvidia_tao_tf2.model_optimization.pruning.pruning import prune

import nvidia_tao_tf2.common.no_warning # noqa pylint: disable=W0611
logger = logging.getLogger(__name__)


class Pruner(ABC):
    """Base Pruner."""

    def __init__(self, cfg) -> None:
        """Initialize."""
        self.cfg = cfg
        self.model_path = cfg.prune.checkpoint
        self.key = cfg.encryption_key
        self.normalizer = cfg.prune.normalizer
        self.criterion = 'L2'
        self.granularity = cfg.prune.granularity
        self.min_num_filters = cfg.prune.min_num_filters
        self.equalization_criterion = cfg.prune.equalization_criterion
        self.excluded_layers = []

    @abstractmethod
    def _load_model(self):
        pass

    def set_model_path(self, model_path):
        """Method to set model path."""
        self.model_path = model_path

    def prune(self, threshold, excluded_layers):
        """Prune a model."""
        self._load_model()
        # Pruning trained model
        pruned_model = prune(
            model=self.model,
            method='min_weight',
            normalizer=self.normalizer,
            criterion=self.criterion,
            granularity=self.granularity,
            min_num_filters=self.min_num_filters,
            threshold=threshold,
            equalization_criterion=self.equalization_criterion,
            excluded_layers=self.excluded_layers + excluded_layers)
        return pruned_model
