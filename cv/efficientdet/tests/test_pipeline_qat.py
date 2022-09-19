# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""EfficientDet QAT pipeline tests."""

import omegaconf
import pytest
import os
import shutil

import tensorflow as tf

from cv.efficientdet.scripts.train import run_experiment as run_train
from cv.efficientdet.scripts.evaluate import run_experiment as run_evaluate
from cv.efficientdet.scripts.export import run_export

TMP_MODEL_DIR = '/home/scratch.metropolis2/tao_ci/tao_tf2/models/tmp'
DATA_DIR = '/home/scratch.metropolis2/tao_ci/tao_tf2/data/coco'


@pytest.fixture(scope='function')
def cfg():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    spec_file = os.path.join(parent_dir, 'experiment_specs', 'default.yaml')
    default_cfg = omegaconf.OmegaConf.load(spec_file)
    default_cfg.data.train_tfrecords = [DATA_DIR + '/val-*']
    default_cfg.data.val_tfrecords = [DATA_DIR + '/val-*']
    default_cfg.data.val_json_file = os.path.join(DATA_DIR, "annotations/instances_val2017.json")

    default_cfg.train.num_examples_per_epoch = 128
    default_cfg.train.checkpoint = ''
    default_cfg.train.checkpoint_period = 1

    default_cfg.evaluate.num_samples = 10
    return default_cfg


# @pytest.mark.script_launch_mode('subprocess')
@pytest.mark.parametrize("amp, qat, batch_size, num_epochs",
                         [(False, True, 4, 1)])
def test_train(amp, qat, batch_size, num_epochs, cfg):
    # env = os.environ.copy()
    results_dir = os.path.join(
        TMP_MODEL_DIR,
        f"exp_b{batch_size}_ep{num_epochs}")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat
    cfg.train.batch_size = batch_size
    cfg.train.results_dir = results_dir

    run_train(cfg, ci_run=True)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs",
                         [(False, True, 4, 1)])
def test_eval(amp, qat, batch_size, num_epochs, cfg):
    # reset graph precision
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat

    cfg.evaluate.model_path = os.path.join(
        TMP_MODEL_DIR,
        f"exp_b{batch_size}_ep{num_epochs}",
        "weights",
        f'efficientdet-d0_00{num_epochs}.eff')
    cfg.evaluate.batch_size = batch_size
    run_evaluate(cfg, ci_run=True)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs, max_bs, dynamic_bs, data_type",
                         [(False, True, 4, 1, 1, True, 'int8')])
def test_export(amp, qat, batch_size, num_epochs, max_bs, dynamic_bs, data_type, cfg):
    # reset graph precision
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat

    cfg.export.data_type = data_type
    cfg.export.model_path = os.path.join(
        TMP_MODEL_DIR,
        f"exp_b{batch_size}_ep{num_epochs}",
        "weights",
        f'efficientdet-d0_00{num_epochs}.eff')
    cfg.export.max_batch_size = max_bs
    cfg.export.dynamic_batch_size = dynamic_bs
    cfg.export.output_path = os.path.join(
        TMP_MODEL_DIR,
        f"exp_b{batch_size}_ep{num_epochs}",
        "weights",
        f'efficientdet-d0_00{num_epochs}.onnx')
    cfg.export.cal_cache_file = os.path.join(
        TMP_MODEL_DIR,
        f"exp_b{batch_size}_ep{num_epochs}",
        "weights",
        f'efficientdet-d0_00{num_epochs}.cal')
    cfg.export.engine_file = os.path.join(
        TMP_MODEL_DIR,
        f"exp_b{batch_size}_ep{num_epochs}",
        "weights",
        f'efficientdet-d0_00{num_epochs}.engine')
    cfg.export.cal_image_dir = os.path.join(DATA_DIR, "raw-data", "test2017")

    run_export(cfg, ci_run=True)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
