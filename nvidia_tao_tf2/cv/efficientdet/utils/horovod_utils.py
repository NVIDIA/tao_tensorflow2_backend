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
"""Horovod utils."""
import logging
import os
import multiprocessing
import tensorflow as tf
import horovod.tensorflow.keras as hvd

from nvidia_tao_tf2.common.utils import set_random_seed
logger = logging.getLogger(__name__)


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


def initialize(cfg, logger, training=True):
    """Initialize training."""
    logger.setLevel(logging.INFO)
    hvd.init()
    use_xla = False
    if training:
        os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
        os.environ['TF_NUM_INTEROP_THREADS'] = str(max(2, (multiprocessing.cpu_count() // hvd.size()) - 2))

    if use_xla:
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
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if training:
        set_random_seed(cfg.train.random_seed + hvd.rank())

    if cfg.train.amp and not cfg.train.qat:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    if is_main_process():
        if not os.path.exists(cfg.results_dir):
            os.makedirs(cfg.results_dir, exist_ok=True)
