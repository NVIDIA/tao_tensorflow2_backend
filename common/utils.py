# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""TAO common utils used across all apps."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
from math import exp, log
import os
import struct
import sys

from eff_tao_encryption.tao_codec import encrypt_stream

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2

import tensorflow as tf
from backbones.utils_tf import swish


ENCRYPTION_OFF = False
reg_dict = {0: None, 1: l1, 2: l2}
ap_mode_dict = {0: "sample", 1: "integrate"}

# Define 1MB for filesize calculation.
MB = 2 << 20

CUSTOM_OBJS = {'swish': swish}


def encode_etlt(tmp_file_name, output_file_name, input_tensor_name, key):
    """Encrypt ETLT model."""
    # Encode temporary uff to output file
    with open(tmp_file_name, "rb") as open_temp_file, \
         open(output_file_name, "wb") as open_encoded_file:
        # TODO: @vpraveen: Remove this hack to support multiple input nodes.
        # This will require an update to tlt_converter and DS. Postponing this for now.
        if isinstance(input_tensor_name, list):
            input_tensor_name = input_tensor_name[0]
        open_encoded_file.write(struct.pack("<i", len(input_tensor_name)))
        open_encoded_file.write(input_tensor_name.encode())
        encrypt_stream(open_temp_file,
                       open_encoded_file,
                       key, encryption=True, rewind=False)


def raise_deprecation_warning(task, subtask, args):
    """Raise a deprecation warning based on the module.

    Args:
        task (str): The TLT task to be deprecated.
        subtask (str): The subtask supported by that task.
        args (list): List of arguments to be appended.

    Raises:
        DeprecationWarning: With the actual command to be run.
    """
    if not isinstance(args, list):
        raise TypeError("There should a list of arguments.")
    args_string = " ".join(args)
    new_command = f"{task} {subtask} {args_string}"
    raise DeprecationWarning(
        f"This command has been deprecated in this version of TLT. Please run \n{new_command}"
    )


def parse_arguments(cl_args, supported_tasks=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('module',
                        default='classification',
                        choices=supported_tasks)
    args, unknown_args = parser.parse_known_args(cl_args)
    args = vars(args)
    return args, unknown_args


def get_num_params(model):
    """Get the number of parameters in a model.

    Args:
        model(keras.model.Model): Model object to run count params.

    Returns:
        num_params(int): Number of parameters in a model. Represented
        in units per million.
    """
    return model.count_params() / 1e6


def get_model_file_size(model_path):
    """Get the size of the model.

    Args:
        model_path (str): UNIX path to the model.

    Returns:
        file_size (float): File size in MB.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file wasn't found at {model_path}")
    file_size = os.path.getsize(model_path) / MB
    return file_size


def setup_keras_backend(training_precision, is_training):
    """Setup Keras-specific backend settings for training or inference.

    Args:
        training_precision: (TrainingPrecision or None) Proto object with FP16/FP32 parameters or
            None. None leaves K.floatx() in its previous setting.
        is_training: (bool) If enabled, Keras is set in training mode.
    """
    # Learning phase of '1' indicates training mode -- important for operations
    # that behave differently at training/test times (e.g. batch normalization)
    if is_training:
        K.set_learning_phase(1)
    else:
        K.set_learning_phase(0)

    # Set training precision, if given. Otherwise leave K.floatx() in its previous setting.
    # K.floatx() determines how Keras creates weights and casts them (Keras default: 'float32').
    if training_precision is not None:
        if training_precision == 'float32':
            K.set_floatx('float32')
        elif training_precision == 'float16':
            K.set_floatx('float16')
        else:
            raise RuntimeError('Invalid training precision selected')


def summary_from_value(tag, value, scope=None):
    """Generate a manual simple summary object with a tag and a value."""
    summary = tf.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    if scope:
        summary_value.tag = f'{scope}/{tag}'
    else:
        summary_value.tag = tag
    return summary


def parse_model_load_from_config(train_config):
    """Parse model loading config from protobuf.

    Input:
        the protobuf config at training_config level.
    Output
        model_path (string): the path of model to be loaded. None if not given
        load_graph (bool): Whether to load whole graph. If False, will need to recompile the model
        reset_optim (bool): Whether to reset optim. This field must be true if load_graph is false.
        initial_epoch (int): the starting epoch number. 0 - based
    """
    load_type = train_config.WhichOneof('load_model')
    if load_type is None:
        return None, False, True, 0
    if load_type == 'resume_model_path':
        try:
            epoch = int(train_config.resume_model_path.split('.')[-2].split('_')[-1])
        except Exception as e:
            raise ValueError("Cannot parse the checkpoint path. Did you rename it?") from e
        return train_config.resume_model_path, True, False, epoch
    if load_type == 'pretrain_model_path':
        return train_config.pretrain_model_path, False, True, 0
    if load_type == 'pruned_model_path':
        return train_config.pruned_model_path, True, True, 0
    raise ValueError("training configuration contains invalid load_model type.")


def check_tf_oom(func):
    """A decorator function to check OOM and raise informative errors."""

    def return_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if type(e) == tf.errors.ResourceExhaustedError:
                logger = logging.getLogger(__name__)
                logger.error(
                    "Ran out of GPU memory, please lower the batch size, use a smaller input "
                    "resolution, use a smaller backbone, or enable model parallelism for "
                    "supported TLT architectures (see TLT documentation)."
                )
                sys.exit(1)
            else:
                # throw out the error as-is if they are not OOM error
                raise e
    return return_func


class StepLRScheduler(keras.callbacks.Callback):
    """Step learning rate annealing schedule.

    This callback implements the step learning rate annnealing schedule according to
    the progress of the current experiment. The training progress is defined as the
    ratio of the current iteration to the maximum iterations. The scheduler adjusts the
    learning rate of the experiment in steps at regular intervals.

    Args:
        base lr: Learning rate at the start of the experiment
        gamma : ratio by which the learning rate reduces at every steps
        step_size : step size as percentage of maximum iterations
        max_iterations : Total number of iterations in the current experiment
                         phase
    """

    def __init__(self, base_lr=1e-2, gamma=0.1, step_size=33, max_iterations=12345):
        """__init__ method."""
        super().__init__()

        if not 0.0 <= step_size <= 100.0:
            raise ValueError('StepLRScheduler does not support a step size < 0.0 or > 100.0')
        if not 0.0 <= gamma <= 1.0:
            raise ValueError('StepLRScheduler does not support gamma < 0.0 or > 1.0')
        self.base_lr = base_lr
        self.gamma = gamma
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """Start of training method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError(
                f'StepLRScheduler does not support a progress value < 0.0 or > 1.0 received ({progress})')

        numsteps = self.max_iterations * self.step_size // 100
        exp_factor = self.global_step / numsteps
        lr = self.base_lr * pow(self.gamma, exp_factor)
        return lr


class MultiGPULearningRateScheduler(keras.callbacks.Callback):
    """Learning rate scheduler implementation.

    Implements https://arxiv.org/pdf/1706.02677.pdf (Accurate, Large Minibatch SGD:
    Training ImageNet in 1 Hour) style learning rate schedule.
    Learning rate scheduler modulates learning rate according to the progress in the
    training experiment. Specifically the training progress is defined as the ratio of
    the current iteration to the maximum iterations. Learning rate scheduler adjusts
    learning rate in the following phases:
        Phase 1: 0.0 <= progress < soft_start:
                 Starting from start_lr linearly increase the learning rate to base_lr.
        Phase 2: at every annealing point, divide learning rate by annealing divider.

    Example:
        ```python
        lrscheduler = MultiGPULearningRateScheduler(
            max_iterations=max_iterations)

        model.fit(X_train, Y_train, callbacks=[lrscheduler])
        ```

    Args:
        max_iterations: Total number of iterations in the experiment.
        start_lr: Learning rate at the beginning. In the paper this is the learning rate used
                  with single GPU training.
        base_lr: Maximum learning rate. In the paper base_lr is set as start_lr * number of
                 GPUs.
        soft_start: The progress at which learning rate achieves base_lr when starting from
                    start_lr. Default value set as in the paper.
        annealing_points: A list of progress values at which learning rate is divided by
                          annealing_divider. Default values set as in the paper.
        annealing_divider: A divider for learning rate applied at each annealing point.
                           Default value set as in the paper.
    """

    def __init__(  # pylint: disable=W0102
            self,
            max_iterations,
            start_lr=3e-4,
            base_lr=5e-4,
            soft_start=0.056,
            annealing_points=[0.33, 0.66, 0.88],
            annealing_divider=10.0):
        """__init__ method."""
        super().__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start varible should be >= 0.0 or <= 1.0.')
        prev = 0.
        for p in annealing_points:
            if not 0.0 <= p <= 1.0:
                raise ValueError('annealing_point should be >= 0.0 or <= 1.0.')
            if p < prev:
                raise ValueError('annealing_points should be in increasing order.')
            if not soft_start < p:
                raise ValueError('soft_start should be less than the first annealing point.')
            prev = p

        self.start_lr = start_lr
        self.base_lr = base_lr
        self.soft_start = soft_start  # Increase to lr from start_lr until this point.
        self.annealing_points = annealing_points  # Divide lr by annealing_divider at these points.
        self.annealing_divider = annealing_divider
        self.max_iterations = max_iterations
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError(
                f'MultiGPULearningRateScheduler does not support a progress value < 0.0 or > 1.0 received ({progress})')

        if not self.base_lr:
            return self.base_lr

        lr = self.base_lr
        if progress < self.soft_start:
            soft_start = progress / self.soft_start
            lr = soft_start * self.base_lr + (1. - soft_start) * self.start_lr
        else:
            for p in self.annealing_points:
                if progress > p:
                    lr /= self.annealing_divider

        return lr


class SoftStartAnnealingLearningRateScheduler(keras.callbacks.Callback):
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

    Example:
        ```python
        lrscheduler = modulus.callbacks.SoftStartAnnealingLearningRateScheduler(
            max_iterations=max_iterations)

        model.fit(X_train, Y_train, callbacks=[lrscheduler])
        ```

    Args:
        base_lr: Maximum learning rate
        min_lr_ratio: The ratio between minimum learning rate (min_lr) and base_lr
        soft_start: The progress at which learning rate achieves base_lr when starting from min_lr
        annealing_start: The progress at which learning rate starts to drop from base_lr to min_lr
        max_iterations: Total number of iterations in the experiment
    """

    def __init__(self, max_iterations, base_lr=5e-4, min_lr_ratio=0.01, soft_start=0.1,
                 annealing_start=0.7):
        """__init__ method."""
        super().__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start varible should be >= 0.0 or <= 1.0.')
        if not 0.0 <= annealing_start <= 1.0:
            raise ValueError('The annealing_start variable should be >= 0.0 or <= 1.0.')
        if not soft_start < annealing_start:
            raise ValueError('Varialbe soft_start should not be less than annealing_start.')

        self.base_lr = base_lr
        self.min_lr_ratio = min_lr_ratio
        self.soft_start = soft_start  # Increase to lr from min_lr until this point.
        self.annealing_start = annealing_start  # Start annealing to min_lr at this point.
        self.max_iterations = max_iterations
        self.min_lr = min_lr_ratio * base_lr
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global_step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not 0. <= progress <= 1.:
            raise ValueError(f'SoftStartAnnealingLearningRateScheduler does not support a progress value < 0.0 or > 1.0 received ({progress})')

        if not self.base_lr:
            return self.base_lr

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

        lr = exp(log(self.min_lr) + t * (log(self.base_lr) - log(self.min_lr)))

        return lr


class OneIndexedCSVLogger(keras.callbacks.CSVLogger):
    """CSV Logger with epoch number started from 1."""

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end."""
        super().on_epoch_end(epoch + 1, logs)


class SoftStartCosineAnnealingScheduler(keras.callbacks.Callback):
    """Soft Start Cosine annealing scheduler.

        learning rate in the following 2 phases:
        Phase 1: 0.0 <= progress < soft_start:
                 Starting from min_lr linearly increase the learning rate to base_lr
        Phase 2: soft_start <= progress <= 1.0:
                 Starting from base_lr cosine decay the learning rate to min_lr

    Args:
        base_lr: Maximum learning rate
        min_lr_ratio: The ratio between minimum learning rate (min_lr) and base_lr
        soft_start: The progress at which learning rate achieves base_lr when starting from min_lr
        max_iterations: Total number of iterations in the experiment

        (https://arxiv.org/pdf/1608.03983.pdf)
    """

    def __init__(self, base_lr, min_lr_ratio, soft_start, max_iterations):
        """Initalize global parameters."""
        super().__init__()

        if not 0.0 <= soft_start <= 1.0:
            raise ValueError('The soft_start varible should be >= 0.0 or <= 1.0.')

        self.max_iterations = max_iterations
        self.soft_start = soft_start
        self.base_lr = base_lr
        self.min_lr = self.base_lr * min_lr_ratio
        self.global_step = 0

    def reset(self, initial_step):
        """Reset global step."""
        self.global_step = initial_step

    def update_global_step(self):
        """Increment global_step by 1."""
        self.global_step += 1

    def on_train_begin(self, logs=None):
        """on_train_begin method."""
        self.reset(self.global_step)
        lr = self.get_learning_rate(self.global_step / float(self.max_iterations))
        K.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        self.update_global_step()
        progress = self.global_step / float(self.max_iterations)
        lr = self.get_learning_rate(progress)
        K.set_value(self.model.optimizer.lr, lr)

    def on_epoch_end(self, epoch, logs):
        """on_epoch_end method."""
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def get_learning_rate(self, progress):
        """Compute learning rate according to progress to reach max_iterations."""
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')

        if not 0. <= progress <= 1.:
            raise ValueError(f'SoftStartCosineAnnealingScheduler does not support a progress value < 0.0 or > 1.0 received ({progress})')

        if not self.base_lr:
            return self.base_lr

        if self.soft_start > 0.0:
            soft_start = progress / self.soft_start
        else:  # learning rate starts from base_lr
            soft_start = 1.0
        if soft_start < 1:
            lr = (self.base_lr - self.min_lr) * soft_start + self.min_lr
        else:
            lr = self.min_lr + (self.base_lr - self.min_lr) * \
                (1 + math.cos(math.pi * (progress - self.soft_start))) / 2
        return lr


class TensorBoard(keras.callbacks.Callback):
    """Callback to log some things to TensorBoard. Quite minimal, and just here as an example."""

    def __init__(self, log_dir='./logs', write_graph=True, weight_hist=False):
        """__init__ method.

        Args:
          log_dir: the path of the directory where to save the log
            files to be parsed by TensorBoard.
          write_graph: whether to visualize the graph in TensorBoard.
            The log file can become quite large when
            write_graph is set to True.
          weight_hist: whether plot histogram of weights.
        """
        super().__init__()
        self.log_dir = log_dir
        self.write_graph = write_graph
        self._merged = None
        self._step = 0
        self._weight_hist = weight_hist

    def on_epoch_begin(self, epoch, logs=None):
        """on_epoch_begin method."""
        # Run user defined summaries
        if self._merged is not None:
            summary_str = self.sess.run(self._merged)
            self.writer.add_summary(summary_str, epoch)
            self.writer.flush()

    def on_batch_end(self, batch, logs=None):
        """on_batch_end method."""
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = summary_from_value(name, value.item())
            self.writer.add_summary(summary, self._step)

        summary = summary_from_value('lr', K.get_value(self.model.optimizer.lr))
        self.writer.add_summary(summary, self._step)

        self._step += 1
        self.writer.flush()

    def set_model(self, model):
        """set_model method."""
        self.model = model
        self.sess = K.get_session()

        if self._weight_hist:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)

        self._merged = tf.summary.merge_all()
        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

    def on_train_end(self, *args, **kwargs):
        """on_train_end method."""
        self.writer.close()
