# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
"""Horovod utils."""
import os
import multiprocessing
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd


def get_rank():
    """Get rank."""
    try:
        return hvd.rank()
    except Exception:
        return 0


def get_world_size():
    """Get world size."""
    try:
        return hvd.size()
    except Exception:
        return 1


def is_main_process():
    """Check if the current process is rank 0."""
    return get_rank() == 0


def initialize(config, training=True):
    """Initialize training."""
    if training and config.set_num_threads:
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = str(max(2, (multiprocessing.cpu_count() // hvd.size()) - 2))

    if config.use_xla:
        # it turns out tf_xla_enable_lazy_compilation is used before importing tersorflow for the first time,
        # so setting this flag in the current function would have no effect. Thus, this flag is already
        # set in Dockerfile. The remaining XLA flags are set here.
        TF_XLA_FLAGS = os.environ['TF_XLA_FLAGS']  # contains tf_xla_enable_lazy_compilation
        os.environ['TF_XLA_FLAGS'] = TF_XLA_FLAGS + " --tf_xla_auto_jit=1"
        os.environ['TF_EXTRA_PTXAS_OPTIONS'] = "-sw200428197=true"
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        assert tf.config.experimental.get_memory_growth(gpu)
    tf.config.experimental.set_visible_devices(gpus, 'GPU')
    if gpus and training:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if training:
        np.random.seed(config.seed)
        tf.random.set_seed(config.seed)

    if config.mixed_precision:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'
