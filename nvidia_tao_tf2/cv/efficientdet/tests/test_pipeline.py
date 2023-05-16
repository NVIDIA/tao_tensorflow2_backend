# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""EfficientDet nonQAT pipeline tests."""

from datetime import datetime
import omegaconf
import pytest
import os
import shutil

import horovod.tensorflow.keras as hvd
import tensorflow as tf

from nvidia_tao_tf2.cv.efficientdet.scripts.train import run_experiment as run_train
from nvidia_tao_tf2.cv.efficientdet.scripts.evaluate import run_experiment as run_evaluate
from nvidia_tao_tf2.cv.efficientdet.scripts.inference import infer_tlt
from nvidia_tao_tf2.cv.efficientdet.scripts.export import run_export
from nvidia_tao_tf2.cv.efficientdet.scripts.prune import run_pruning

TMP_MODEL_DIR = '/home/scratch.metropolis2/tao_ci/tao_tf2/models/tmp'
DATA_DIR = '/home/scratch.metropolis2/tao_ci/tao_tf2/data/coco'
time_str = datetime.now().strftime("%y_%m_%d_%H:%M:%S")
hvd.init()


@pytest.fixture(scope='function')
def cfg():
    parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    spec_file = os.path.join(parent_dir, 'experiment_specs', 'default.yaml')
    default_cfg = omegaconf.OmegaConf.load(spec_file)
    default_cfg.dataset.train_tfrecords = [DATA_DIR + '/val-*']
    default_cfg.dataset.val_tfrecords = [DATA_DIR + '/val-*']
    default_cfg.dataset.val_json_file = os.path.join(DATA_DIR, "annotations/instances_val2017.json")

    default_cfg.train.num_examples_per_epoch = 128
    default_cfg.train.checkpoint = ''
    default_cfg.train.checkpoint_interval = 1

    default_cfg.evaluate.num_samples = 10
    return default_cfg


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs",
                         [(True, False, 2, 2)])
def test_train(amp, qat, batch_size, num_epochs, cfg):
    results_dir = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat
    cfg.train.batch_size = batch_size
    cfg.results_dir = results_dir

    run_train(cfg)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs",
                         [(True, False, 2, 2)])
def test_eval(amp, qat, batch_size, num_epochs, cfg):
    # reset graph precision
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat

    cfg.evaluate.checkpoint = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}",
        f'efficientdet-d0_00{num_epochs}.tlt')
    cfg.evaluate.batch_size = batch_size
    run_evaluate(cfg)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs, max_bs, dynamic_bs, data_type",
                         [(True, False, 2, 2, 1, True, 'int8')])
def test_export(amp, qat, batch_size, num_epochs, max_bs, dynamic_bs, data_type, cfg):
    # reset graph precision
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat

    cfg.export.checkpoint = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}",
        f'efficientdet-d0_00{num_epochs}.tlt')
    cfg.export.batch_size = max_bs
    cfg.export.dynamic_batch_size = dynamic_bs
    cfg.export.onnx_file = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}",
        f'efficientdet-d0_00{num_epochs}.onnx')

    run_export(cfg)
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs",
                         [(True, False, 2, 2)])
def test_infer(amp, qat, batch_size, num_epochs, cfg):

    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat

    cfg.inference.image_dir = os.path.join(DATA_DIR, "raw-data", "debug2017")
    cfg.inference.output_dir = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}",
        'infer_output')
    cfg.inference.checkpoint = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}",
        f'efficientdet-d0_00{num_epochs}.tlt')
    infer_tlt(cfg)


@pytest.mark.parametrize("amp, qat, batch_size, num_epochs",
                         [(True, False, 2, 2)])
def test_prune(amp, qat, batch_size, num_epochs, cfg):
    cfg.train.num_epochs = num_epochs
    cfg.train.amp = amp
    cfg.train.qat = qat
    cfg.prune.checkpoint = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}",
        f'efficientdet-d0_00{num_epochs}.tlt')
    cfg.prune.results_dir = os.path.join(
        TMP_MODEL_DIR,
        f"effdet_b{batch_size}_ep{num_epochs}_{time_str}")
    run_pruning(cfg)
