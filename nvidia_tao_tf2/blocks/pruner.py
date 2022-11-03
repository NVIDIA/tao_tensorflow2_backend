# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO classification model pruner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
        self.model_path = cfg.prune.model_path
        self.key = cfg.key
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
