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
"""Weighted fusion layer."""

import tensorflow as tf


class WeightedFusion(tf.keras.layers.Layer):
    """Weighted Fusion Layer (Fast Attention)."""

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

    def call(self, inputs):
        """Call."""
        dtype = inputs[0].dtype
        edge_weights = []
        for var in self.vars:
            var = tf.nn.relu(tf.cast(var, dtype=dtype))
            edge_weights.append(var)
        weights_sum = tf.add_n(edge_weights)
        inputs = [
            inputs[i] * edge_weights[i] / (weights_sum + 1e-4)
            for i in range(len(inputs))
        ]
        new_node = tf.add_n(inputs)
        return new_node

    def get_config(self):
        """Config."""
        config = super().get_config()
        config.update({
            'inputs_offsets': self.inputs_offsets
        })
        return config
