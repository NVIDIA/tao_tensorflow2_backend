# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Tensorboard callback for learning rate schedules."""

import tensorflow as tf


class LRTensorBoard(tf.keras.callbacks.Callback):
    """Learning Rate Tensorboard Callback."""

    def __init__(self, log_dir, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.steps_before_epoch = 0
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
