# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO classification model pruner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from abc import ABC, abstractmethod
from model_optimization.pruning.pruning import prune

import common.no_warning # noqa pylint: disable=W0611
logger = logging.getLogger(__name__)


class Pruner(ABC):
    
    def __init__(self, model_path, key,
                 normalizer='max',
                 criterion='L2',
                 granularity=8,
                 min_num_filters=16,
                 equalization_criterion='union') -> None:
        self.model_path = model_path
        self.key = key
        self.normalizer = normalizer
        self.criterion = criterion
        self.granularity = granularity
        self.min_num_filters = min_num_filters
        self.equalization_criterion = equalization_criterion
        self.excluded_layers = []
    
    @abstractmethod
    def _load_model(self):
        pass
        
    def set_model_path(self, model_path):
        self.model_path = model_path

    def prune(self, threshold, excluded_layers, verbosity='DEBUG'):
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

        if verbosity == 'DEBUG':
            # Printing out pruned model summary
            logger.debug("Model summary of the pruned model:")
            logger.debug(pruned_model.summary())

        pruning_ratio = pruned_model.count_params() / self.model.count_params()
        logger.info(
            "Pruning ratio (pruned model / original model): {}".format(
                pruning_ratio
            )
        )
        return pruned_model
