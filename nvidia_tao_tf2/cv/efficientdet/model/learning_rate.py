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
"""Learning rate related utils."""
import math
import logging
from typing import Any, Mapping
import tensorflow as tf
logger = logging.getLogger(__name__)


@tf.keras.utils.register_keras_serializable(package='Custom')
class CosineLrSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate schedule."""

    def __init__(self, base_lr: float, lr_warmup_init: float,
                 lr_warmup_step: int, total_steps: int):
        """Build a CosineLrSchedule.

        Args:
            base_lr: `float`, The initial learning rate.
            lr_warmup_init: `float`, The warm up learning rate.
            lr_warmup_step: `int`, The warm up step.
            total_steps: `int`, Total train steps.
        """
        super().__init__()
        logger.debug('LR schedule method: cosine')
        self.base_lr = base_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

    def __call__(self, step):
        """Call."""
        linear_warmup = (
            self.lr_warmup_init +
            (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
                (self.base_lr - self.lr_warmup_init)))
        cosine_lr = 0.5 * self.base_lr * (
            1 + tf.cos(math.pi * (tf.cast(step, tf.float32) - self.lr_warmup_step) / self.decay_steps))
        return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)

    def get_config(self) -> Mapping[str, Any]:
        """Get config."""
        return {
            "base_lr": self.base_lr,
            "lr_warmup_init": self.lr_warmup_init,
            "lr_warmup_step": self.lr_warmup_step,
        }


@tf.keras.utils.register_keras_serializable(package='Custom')
class SoftStartAnnealingLearningRateScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate scheduler implementation.

    Learning rate scheduler modulates learning rate according to the progress in the
    training experiment. Specifically the training progress is defined as the ratio of
    the current iteration to the maximum iterations. Learning rate scheduler adjusts
    learning rate in the following 3 phases:
            Phase 1: 0.0 <= progress < soft_start:
                                Starting from min_lr exponentially increase the learning rate to base_lr
            Phase 2: soft_start <= progress < annealing_start:
                                Maintain the learning rate at base_lr
            Phase 3: annealing_start <= progress <= 1.0:
                                Starting from base_lr exponentially decay the learning rate to min_lr

    Args:
            base_lr: Maximum learning rate
            min_lr_ratio: The ratio between minimum learning rate (min_lr) and base_lr
            soft_start: The progress at which learning rate achieves base_lr when starting from min_lr
            annealing_start: The progress at which learning rate starts to drop from base_lr to min_lr
            total_steps: Total number of iterations in the experiment
    """

    def __init__(self, base_lr, lr_warmup_init, soft_start,
                 annealing_start, total_steps):
        """__init__ method."""
        super().__init__()
        logger.debug('LR schedule method: SoftStartAnnealing')
        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start variable should be >= 0.0 or <= 1.0.')
        if not 0.0 <= annealing_start <= 1.0:
            raise ValueError('The annealing_start variable should be >= 0.0 or <= 1.0.')
        if not soft_start < annealing_start:
            raise ValueError('Variable soft_start should be less than annealing_start.')

        self.base_lr = base_lr
        self.soft_start = soft_start  # Increase to lr from min_lr until this point.
        self.annealing_start = annealing_start  # Start annealing to min_lr at this point.
        self.total_steps = total_steps
        self.lr_warmup_init = lr_warmup_init

    def __call__(self, step):
        """Compute learning rate according to progress to reach total_steps."""
        progress = step / self.total_steps
        if self.soft_start > 0.0:
            soft_start = progress / self.soft_start
        else:  # learning rate starts from base_lr
            soft_start = 1.0

        if self.annealing_start < 1.0:
            annealing = (1.0 - progress) / (1.0 - self.annealing_start)
        else:
            annealing = 1.0

        # t = soft_start if progress < self.soft_start else 1.0
        t = tf.where(progress < self.soft_start, soft_start, 1.0)
        # t = annealing if progress > self.annealing_start else t
        t = tf.where(progress > self.annealing_start, annealing, t)
        t = tf.cast(t, dtype=tf.float32)
        lr = tf.math.exp(tf.math.log(self.lr_warmup_init) +
                         t * (tf.math.log(self.base_lr) - tf.math.log(self.lr_warmup_init)))
        return lr

    def get_config(self):
        """Config."""
        return {
            "base_lr": self.base_lr,
            "lr_warmup_init": self.lr_warmup_init,
            "soft_start": self.soft_start,
            "annealing_start": self.annealing_start,
        }


def learning_rate_schedule(params, steps_per_epoch):
    """Learning rate schedule based on global step.

    Args:
        params (TrainConfig): train config loaded by Hydra.
        steps_per_epoch (int): Number of steps per epoch.
    """
    supported_schedules = ['cosine', 'soft_anneal']
    lr_warmup_step = int(params.lr_warmup_epoch * steps_per_epoch)
    total_steps = int(params.num_epochs * steps_per_epoch)
    lr_decay_method = str(params.lr_decay_method)
    if lr_decay_method == 'cosine':
        return CosineLrSchedule(params.learning_rate,
                                params.lr_warmup_init, lr_warmup_step,
                                total_steps)
    if lr_decay_method == 'soft_anneal':
        return SoftStartAnnealingLearningRateScheduler(
            params.learning_rate,
            params.lr_warmup_init,
            params.lr_warmup_epoch / params.num_epochs,
            params.lr_annealing_epoch / params.num_epochs,
            total_steps)

    raise ValueError(f'unknown lr_decay_method: {lr_decay_method}. \
        Choose from {supported_schedules}')
