# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""EfficientDet model tests."""

import omegaconf
import pytest
import os
import numpy as np

import tensorflow as tf

from nvidia_tao_tf2.cv.efficientdet.model.efficientdet import efficientdet
from nvidia_tao_tf2.cv.efficientdet.model.model_builder import build_backbone
from nvidia_tao_tf2.cv.efficientdet.utils import hparams_config
from nvidia_tao_tf2.cv.efficientdet.utils.config_utils import generate_params_from_cfg
from nvidia_tao_tf2.cv.efficientdet.utils.model_utils import num_params


@pytest.fixture(scope='function')
def cfg():
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    spec_file = os.path.join(parent_dir, 'experiment_specs', 'default.yaml')
    default_cfg = omegaconf.OmegaConf.load(spec_file)
    default_cfg.data_format = 'channels_last'
    default_cfg.train.num_examples_per_epoch = 128
    default_cfg.train.checkpoint = ''
    default_cfg.train.checkpoint_interval = 1

    default_cfg.evaluate.num_samples = 10
    return default_cfg


@pytest.mark.parametrize(
    "model_name, input_size, expected",
    [('efficientdet-d0', '512x512', 3880652),
     ('efficientdet-d0', '511x513', 3880652),
     ('efficientdet-d1', '128x512', 6626699),
     ('efficientdet-d2', '512x512', 8098056),
     ('efficientdet-d3', '256x512', 12033745),
     ('efficientdet-d4', '512x256', 20725700),
     ('efficientdet-d5', '512x128', 33655916),])
def test_arch(model_name, input_size, expected, cfg):
    cfg.data.image_size = input_size
    cfg.model.name = model_name
    config = hparams_config.get_efficientdet_config(model_name)
    config.update(generate_params_from_cfg(config, cfg, mode='train'))
    input_shape = list(config.image_size) + [3]
    model = efficientdet(input_shape, training=True, config=config)
    assert num_params(model) == expected
    tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize(
    "model_name, input_size",
    [('efficientdet-d0', '512x512',),
     ('efficientdet-d0', '512x128',),
     ('efficientdet-d1', '128x512',),
     ('efficientdet-d2', '512x512',),
     ('efficientdet-d3', '256x512',),
     ('efficientdet-d4', '512x256',),
     ('efficientdet-d5', '512x128',),])
def test_backbone(model_name, input_size, cfg):
    cfg.data.image_size = input_size
    cfg.model.name = model_name
    config = hparams_config.get_efficientdet_config(model_name)
    config.update(generate_params_from_cfg(config, cfg, mode='train'))
    input_shape = list(config.image_size) + [3]
    inputs = tf.keras.Input(shape=input_shape)
    features = build_backbone(inputs, config)
    np.testing.assert_array_equal(list(features.keys()), [0, 1, 2, 3, 4, 5])
    np.testing.assert_array_equal(features[0].shape[1:3], list(config.image_size))
    np.testing.assert_array_equal(features[5].shape[1:3], list(map(lambda x: x // 32, config.image_size)))
    tf.compat.v1.reset_default_graph()
