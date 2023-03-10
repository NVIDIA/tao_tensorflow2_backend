# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""EfficientDet utils tests."""

from functools import partial
import math
import numpy as np
import pytest

import tensorflow as tf

from nvidia_tao_tf2.cv.efficientdet.model.activation_builder import activation_fn


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softplus(x):
    return math.log(math.exp(x) + 1)


def expected_result(value, type):
    if type in ['silu', 'swish']:
        return sigmoid(value) * value
    elif type == 'hswish':
        return value * min(max(0, value + 3), 6) / 6
    elif type == 'mish':
        return math.tanh(softplus(value)) * value
    raise ValueError(f"{type} not supported")


@pytest.mark.parametrize(
    "type",
    ['silu', 'swish', 'hswish', 'mish'])
def test_activations(type):
    values = [.5, 10]
    inputs = tf.constant(values)
    outputs = activation_fn(inputs, type)
    _expected = partial(expected_result, type=type)
    assert np.allclose(outputs.numpy(), list(map(lambda x: _expected(x), values)))
