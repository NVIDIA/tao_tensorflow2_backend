# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Custom CSV Logger."""

from tensorflow.keras.callbacks import CSVLogger

import nvidia_tao_tf2.common.logging.logging as status_logging


class CSVLoggerWithStatus(CSVLogger):
    """Callback that streams epoch results to a CSV file and status logger.

    Args:
        filename: Filename of the CSV file, e.g. `'run/log.csv'`.
        separator: String used to separate elements in the CSV file.
        append: Boolean. True: append if file exists (useful for continuing
            training). False: overwrite existing file.
    """

    def on_epoch_end(self, epoch, logs=None):
        """Add metrics to status logger on epoch end."""
        super().on_epoch_end(epoch, logs)
        for key in self.keys:
            status_logging.get_status_logger().kpi[key] = float(logs[key])
