# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Metric logging callback."""

from datetime import timedelta
import numpy as np
import time
import tensorflow as tf

from common.mlops.wandb import alert
import common.logging.logging as status_logging


class MetricLogging(tf.keras.callbacks.Callback):
    """Learning Rate Tensorboard Callback."""

    def __init__(self, num_epochs, steps_per_epoch, initial_epoch, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.steps_before_epoch = steps_per_epoch * initial_epoch
        self.steps_in_epoch = 0
        # Initialize variables for epoch time calculation.
        self.time_per_epoch = 0
        # total loss
        self.total_loss = 0
        self.s_logger = status_logging.get_status_logger()

    @property
    def global_steps(self):
        """The current 1-indexed global step."""
        return self.steps_before_epoch + self.steps_in_epoch

    def on_batch_end(self, batch, logs=None):
        """on_batch_end."""
        self.steps_in_epoch = batch + 1
        if np.isnan(float(logs.get('loss'))):
            alert(
                title='nan loss',
                text='Training loss is nan',
                level=1,
                duration=1800
            )
        self.total_loss += float(logs.get('loss'))

    def on_epoch_begin(self, epoch, logs=None):
        """on_epoch_begin."""
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """on_epoch_end."""
        self.steps_before_epoch += self.steps_in_epoch
        avg_loss = self.total_loss / self.steps_in_epoch
        epoch_end_time = time.time()
        self.time_per_epoch = epoch_end_time - self._epoch_start_time
        # reset loss and steps
        self.steps_in_epoch = 0
        self.total_loss = 0
        # dump log
        self.write_status_json(avg_loss, epoch)

    def write_status_json(self, loss, current_epoch):
        """Write out the data to the status.json file initiated by the experiment for monitoring.

        Args:
            loss (float): average training loss to be recorder in the monitor.
            current_epoch (int): Current epoch.
        """
        current_epoch += 1  # 1-based
        lr = self.model.optimizer.lr(self.global_steps).numpy()
        monitor_data = {
            "epoch": current_epoch,
            "max_epoch": self.num_epochs,
            "time_per_epoch": str(timedelta(seconds=self.time_per_epoch)),
            # "eta": str(timedelta(seconds=(self.num_epochs - current_epoch) * self.time_per_epoch)),
            "loss": loss,
            "learning_rate": float(lr)
        }
        # Save the json file.
        try:
            self.s_logger.write(
                data=monitor_data,
                status_level=status_logging.Status.RUNNING)
        except IOError:
            # We let this pass because we do not want the json file writing to crash the whole job.
            pass
