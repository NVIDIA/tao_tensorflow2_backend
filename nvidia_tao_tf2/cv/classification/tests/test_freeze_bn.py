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
# limitations under the License."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import pytest

from nvidia_tao_tf2.cv.classification.model.model_builder import get_model
from nvidia_tao_tf2.cv.classification.utils.helper import setup_config


class RegConfig():
    """Class for reg config."""

    def __init__(self, reg_type, scope, weight_decay):
        self.type = reg_type
        self.scope = scope
        self.weight_decay = weight_decay


bn_config = (False, True)


@pytest.mark.parametrize("freeze_bn",
                         bn_config)
def test_freeze_bn(freeze_bn):
    keras.backend.clear_session()
    model = get_model(
        "resnet_10",
        input_shape=(3, 224, 224),
        data_format="channels_first",
        freeze_bn=freeze_bn,
        nlayers=18,
        use_batch_norm=True,
        use_pooling=False,
        dropout=0.0,
        use_bias=False,
        all_projections=False,
    )
    reg_config = RegConfig("L2", "Conv2D,Dense", 1e-5)
    model = setup_config(
        model,
        reg_config.__dict__,
    )
    model.compile(
        loss="mse",
        metrics=['accuracy'],
        optimizer="sgd"
    )
    if freeze_bn:
        assert check_freeze_bn(model), (
            "BN layers not frozen, expected frozen."
        )
    else:
        assert not check_freeze_bn(model), (
            "BN layers frozen, expected not frozen."
        )


def check_freeze_bn(model):
    """Check if the BN layers in a model is frozen or not."""
    bn_weights = []
    for layer in model.layers:
        if type(layer) == BatchNormalization:
            # only check for moving mean and moving variance
            bn_weights.append(layer.get_weights()[2:])
    rand_input = np.random.random((1, 3, 224, 224))
    # do training several times
    out_shape = model.outputs[0].get_shape()[1:]
    out_label = np.random.random((1,) + out_shape)
    model.train_on_batch(rand_input, out_label)
    model.train_on_batch(rand_input, out_label)
    model.train_on_batch(rand_input, out_label)
    # do prediction several times
    model.predict(rand_input)
    model.predict(rand_input)
    model.predict(rand_input)
    # finally, check BN weights
    new_bn_weights = []
    for layer in model.layers:
        if type(layer) == BatchNormalization:
            # only check for moving mean and moving variance
            new_bn_weights.append(layer.get_weights()[2:])
    # check the bn weights
    for old_w, new_w in zip(bn_weights, new_bn_weights):
        if not np.array_equal(old_w, new_w):
            return False
    return True
