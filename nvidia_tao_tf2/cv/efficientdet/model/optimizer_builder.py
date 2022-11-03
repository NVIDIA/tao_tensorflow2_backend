# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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
# ==============================================================================
"""Optimizer related utils."""
import logging
import tensorflow as tf
from tensorflow_addons.optimizers import MovingAverage

from nvidia_tao_tf2.cv.efficientdet.model import learning_rate


@tf.keras.utils.register_keras_serializable(package='Custom')
class HvdMovingAverage(MovingAverage):
    """MovingAverage to support Horovod."""

    def swap_weights(self):
        """Swap the average and moving weights.

        The original function in the parent class assumes a cross replica
        context, which fails for single GPU training. It also failed in the case of
        multi-GPU training with Horovod.
        """
        self._swap_weights()

    def _create_slots(self, var_list):
        """HvdMovingAverage _create_slots summary.

        The original function in the parent class, in addition to calling
        _create_slots() of the base optimizer, reassigns trainable tensors to
        self._average_weights and self._model_weights, which has the effect of
        removing non-trainable tensors (e.g., moving means and variances) from EMA.
        By overriding it, we simply keep the part that calls _create_slots of the base
        optimizer. To make up for the removed part of the code, we call shadow_copy, which
        assigns both trainable and non-trainable tensors to self._average_weights and
        self._model_weights.
        Args:
            var_list ([type]): [description]
        """
        self._optimizer._create_slots(var_list=var_list)

    def apply_gradients(self, grads_and_vars, name=None, experimental_aggregate_gradients=True):
        """Apply gradients."""
        self._optimizer._iterations = self.iterations
        result = super().apply_gradients(grads_and_vars, name)
        # update EMA weights after the weights are updated
        self.update_average(self._optimizer.iterations)
        return result

    def _resource_apply_dense(self, grad, var):
        """HvdMovingAverage _resource_apply_dense summary.

        We must override this function, eliminating the part that performs
        EMA updates for trainable variables. The reasons is that we use our custom
        self.update_average(), called in apply_gradients, which performs EMA updates
        for both trainable and non-trainable variables. If we don't override this
        function, in each iteration, EMA of trainable variables get updated twice
        (once here and once in apply_gradient) while EMA of non-trainable variables get
        updated only once in apply_gradients.
        """
        return self._optimizer._resource_apply_dense(grad, var)

    def _resource_apply_sparse(self, grad, var, indices):
        """HvdMovingAverage _resource_apply_sparse summary.

        We must override this function, eliminating the part that performs
        EMA updates for trainable variables. The reasons is that we use our custom
        self.update_average(), called in apply_gradients, which performs EMA updates
        for both trainable and non-trainable variables. If we don't override this
        function, in each iteration, EMA of trainable variables get updated twice
        (once here and once in apply_gradient) while EMA of non-trainable variables get
        updated only once in apply_gradients.
        """
        return self._optimizer._resource_apply_sparse(grad, var, indices)

    def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
        """_resource_apply_sparse_duplicate_indices.

        We must override this function, eliminating the part that performs
        EMA updates for trainable variables. The reasons is that we use our custom
        self.update_average(), called in apply_gradients, which performs EMA updates
        for both trainable and non-trainable variables. If we don't override this
        function, in each iteration, EMA of trainable variables get updated twice
        (once here and once in apply_gradient) while EMA of non-trainable variables get
        updated only once in apply_gradients.
        """
        return self._optimizer._resource_apply_sparse_duplicate_indices(
            grad, var, indices)

    def get_decay_value(self, step, average_decay):
        """Get decay value for updating average."""
        # Pyarmor doesn't recognize the tf.function decorator
        # Which results in the error: using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution.
        # Hence we use tf.py_function as a temporary workaround
        if step < self._start_step:
            decay = tf.constant(0., tf.float32)
        elif self._dynamic_decay:
            decay = step - self._start_step
            decay = tf.minimum(average_decay, (1. + decay) / (10. + decay))
        else:
            decay = average_decay
        return decay

    def update_average(self, step: tf.Tensor):
        """Update average."""
        step = tf.cast(step, tf.float32)
        average_decay = self._get_hyper("average_decay", tf.dtypes.float32)
        decay = tf.py_function(func=self.get_decay_value, inp=[step, average_decay], Tout=average_decay.dtype)

        def _apply_moving(v_moving, v_normal):
            diff = v_moving - v_normal
            v_moving.assign_sub(tf.cast(1. - decay, v_moving.dtype) * diff)
            return v_moving

        def _update(strategy, v_moving_and_v_normal):
            for v_moving, v_normal in v_moving_and_v_normal:
                strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))

        ctx = tf.distribute.get_replica_context()
        return ctx.merge_call(_update, args=(zip(self._average_weights, self._model_weights),))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """From config."""
        optimizer = tf.keras.optimizers.deserialize(
            config.pop('optimizer'),
            custom_objects=custom_objects,
        )
        return cls(optimizer, **config)


def get_optimizer(params, steps_per_epoch):
    """Get optimizer.

    Args:
        params (TrainConfig): train config loaded by Hydra.
        steps_per_epoch (int): Number of steps per epoch.
    """
    lr = learning_rate.learning_rate_schedule(params, steps_per_epoch)
    if params.optimizer.name.lower() == 'sgd':
        logging.info('Use SGD optimizer')
        optimizer = tf.keras.optimizers.SGD(
            lr, momentum=params.optimizer.momentum)
    else:
        raise ValueError('optimizer should be sgd')

    moving_average_decay = params.moving_average_decay
    if moving_average_decay is not None and moving_average_decay > 0.0:
        optimizer = HvdMovingAverage(optimizer, average_decay=moving_average_decay, dynamic_decay=True)

    if params.get('mixed_precision', None):
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, initial_scale=params.get('loss_scale', None))

    return optimizer
