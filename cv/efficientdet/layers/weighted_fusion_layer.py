# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Weighted fusion layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class WeightedFusion(tf.keras.layers.Layer):
    """Weighted Fusion Layer."""

    def __init__(self, inputs_offsets=None, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.inputs_offsets = inputs_offsets
        self.vars = []

    def build(self, input_shape=None):
        """Build."""
        for i, _ in enumerate(self.inputs_offsets):
            name = 'WSM' + ('' if i == 0 else '_' + str(i))
            self.vars.append(self.add_weight(initializer='ones', name=name))

    def call(self, nodes):
        """Call."""
        dtype = nodes[0].dtype
        edge_weights = []
        for var in self.vars:
            var = tf.cast(var, dtype=dtype)
            edge_weights.append(var)
        weights_sum = tf.add_n(edge_weights)
        nodes = [
            nodes[i] * edge_weights[i] / (weights_sum + 1e-4)
            for i in range(len(nodes))
        ]
        new_node = tf.add_n(nodes)
        return new_node

    def get_config(self):
        """Config."""
        config = super().get_config()
        config.update({
            'inputs_offsets': self.inputs_offsets
        })
        return config
