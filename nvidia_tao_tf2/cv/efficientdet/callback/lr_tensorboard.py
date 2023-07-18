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

"""Tensorboard callback for learning rate schedules."""

import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.Callback):
    """Learning Rate Tensorboard Callback."""

    def __init__(self, steps_per_epoch, initial_epoch, log_dir, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.steps_before_epoch = steps_per_epoch * initial_epoch
        self.steps_in_epoch = 0

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_batch_end(self, batch, logs=None):
        """on_batch_end."""
        self.steps_in_epoch = batch + 1

        lr = self.model.optimizer.lr(self.global_steps)
        with self.summary_writer.as_default():
            tf.summary.scalar('learning_rate', lr, self.global_steps)

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end."""
        self.steps_before_epoch += self.steps_in_epoch
        self.steps_in_epoch = 0

    def on_train_end(self, logs=None):
        """on_train_end."""
        self.summary_writer.flush()
