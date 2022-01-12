# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
"""Weighted fusion layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class WeightedFusion(tf.keras.layers.Layer):
    """Weighted Fusion Layer."""

    def __init__(self, epsilon=1e-4, **kwargs):
        """Init."""
        super(WeightedFusion, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        """Build."""
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        """Call."""
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        """Compute output shape."""
        return input_shape[0]

    def get_config(self):
        """Config."""
        config = super(WeightedFusion, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config
