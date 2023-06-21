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

"""Moving average callback implementation."""

import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage
from typing import Any, MutableMapping, Text

from nvidia_tao_tf2.cv.efficientdet.utils.helper import fetch_optimizer


class MovingAverageCallback(tf.keras.callbacks.Callback):
    """A Callback to be used with a `MovingAverage` optimizer.

    Applies moving average weights to the model during validation time to test
    and predict on the averaged weights rather than the current model weights.
    Once training is complete, the model weights will be overwritten with the
    averaged weights (by default).

    Attributes:
        overwrite_weights_on_train_end: Whether to overwrite the current model
            weights with the averaged weights from the moving average optimizer.
        **kwargs: Any additional callback arguments.
    """

    def __init__(self,
                 overwrite_weights_on_train_end: bool = False,
                 **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.overwrite_weights_on_train_end = overwrite_weights_on_train_end
        self.ema_opt = None

    def set_model(self, model: tf.keras.Model):
        """Set model."""
        super().set_model(model)
        self.ema_opt = fetch_optimizer(model, MovingAverage)
        self.ema_opt.shadow_copy(self.model.weights)

    def on_test_begin(self, logs: MutableMapping[Text, Any] = None):
        """Override on_step_begin."""
        self.ema_opt.swap_weights()

    def on_test_end(self, logs: MutableMapping[Text, Any] = None):
        """Override on_test_end."""
        self.ema_opt.swap_weights()

    def on_train_end(self, logs: MutableMapping[Text, Any] = None):
        """Override on_train_end."""
        if self.overwrite_weights_on_train_end:
            self.ema_opt.assign_average_vars(self.model.variables)
