# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO classification model pruner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from blocks.pruner import Pruner
from common.utils import CUSTOM_OBJS
from cv.classification.utils.helper import decode_tltb, decode_eff
from model_optimization.pruning.pruning import prune

from tensorflow import keras
import common.no_warning # noqa pylint: disable=W0611
logger = logging.getLogger(__name__)


class ClassificationPruner(Pruner):
    """Classification pruner class."""

    def _load_model(self):
        """Load classification model."""
        self.model_path = decode_eff(self.model_path, self.key)
        if self.cfg.prune.byom_model_path:
            custom_objs = decode_tltb(self.cfg.prune.byom_model_path, self.key)['custom_objs']
            CUSTOM_OBJS.update(custom_objs)

        # @scha: Although TF SavedModel can mostly train / eval
        # models with custom layer w/o the actual implementation,
        # pruning require layer configuration. Hence, better to
        # specify custom objects while loading
        self.model = keras.models.load_model(
            self.model_path, custom_objects=CUSTOM_OBJS
        )
        self.excluded_layers = ['predictions', 'predictions_dense']
        self.model.summary()

    def _handle_byom_layers(self, excluded_layers):
        """Handle BYOM custom layers."""
        byom_custom_layer = set()
        # For BYOM Models with custom layer
        for layer in self.model.layers:
            # Custom layers are automatically added to excluded_layers
            # and byom_custom_layer which will automatically update
            # TRAVERSABLE_LAYERS in model_optimization/pruning/pruning.py
            if 'helper' in str(type(layer)):
                excluded_layers.append(layer.name)
                byom_custom_layer.add(type(layer))

            # Lambda layers in BYOM models are automatically excluded.
            if type(layer) == keras.layers.Lambda:
                excluded_layers.append(layer.name)
        byom_custom_layer = list(byom_custom_layer)
        return byom_custom_layer, excluded_layers

    def prune(self, threshold, excluded_layers):
        """Run pruning."""
        self._load_model()
        byom_custom_layer = None
        if self.cfg.prune.byom_model_path:
            logger.info("Loading BYOM information")
            byom_custom_layer, excluded_layers = self._handle_byom_layers(excluded_layers)

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
            excluded_layers=self.excluded_layers + excluded_layers,
            byom_custom_layer=byom_custom_layer)

        return pruned_model
