# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
"""Learning rate related utils."""
import math
from absl import logging
from typing import Any, Mapping
import tensorflow as tf


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
    super(CosineLrSchedule, self).__init__()
    logging.info('LR schedule method: cosine')
    self.base_lr = base_lr
    self.lr_warmup_init = lr_warmup_init
    self.lr_warmup_step = lr_warmup_step
    self.decay_steps = tf.cast(total_steps - lr_warmup_step, tf.float32)

  def __call__(self, step):
    linear_warmup = (
        self.lr_warmup_init +
        (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
         (self.base_lr - self.lr_warmup_init)))
    cosine_lr = 0.5 * self.base_lr * (
        1 + tf.cos(math.pi * (tf.cast(step, tf.float32) - self.lr_warmup_step) / self.decay_steps))
    return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)

  def get_config(self) -> Mapping[str, Any]:
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
        super(SoftStartAnnealingLearningRateScheduler, self).__init__()

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
        progress = tf.cast(step / self.total_steps, dtype=tf.float32)
        if not 0. <= progress <= 1.:
            raise ValueError('SoftStartAnnealingLearningRateScheduler '
                             'does not support a progress value < 0.0 or > 1.0 '
                             'received (%f)' % progress)

        if self.soft_start > 0.0:
            soft_start = progress / self.soft_start
        else:  # learning rate starts from base_lr
            soft_start = 1.0

        if self.annealing_start < 1.0:
            annealing = (1.0 - progress) / (1.0 - self.annealing_start)
        else:   # learning rate is never annealed
            annealing = 1.0

        t = soft_start if progress < self.soft_start else 1.0
        t = annealing if progress > self.annealing_start else t
        t = tf.cast(t, dtype=tf.float32)
        lr = exp(log(self.lr_warmup_init) + t * (log(self.base_lr) - log(self.lr_warmup_init)))
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
  """Learning rate schedule based on global step."""
  supported_schedules = ['cosine', 'soft_anneal']
  lr_warmup_step = int(params['lr_warmup_epoch'] * steps_per_epoch)
  total_steps = int(params['num_epochs'] * steps_per_epoch)
  lr_decay_method = params['lr_decay_method']
  if lr_decay_method == 'cosine':
    return CosineLrSchedule(params['learning_rate'],
                            params['lr_warmup_init'], lr_warmup_step,
                            total_steps)
  if lr_decay_method == 'soft_anneal':
    return SoftStartAnnealingLearningRateScheduler(
      total_steps,
      params['learning_rate'],
      params['lr_warmup_init'],
      0.1, 0.3, # TODO(@yuw): add config
      total_steps)

  raise ValueError(f'unknown lr_decay_method: {lr_decay_method}.
    Choose from {supported_schedules}')
