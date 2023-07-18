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
"""Custom CSV Logger."""
from datetime import timedelta
import time
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

    def __init__(self, *args, **kwargs) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.s_logger = status_logging.get_status_logger()

    def on_epoch_begin(self, epoch, logs=None):
        """on_epoch_begin."""
        self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Add metrics to status logger on epoch end."""
        epoch = epoch + 1
        super().on_epoch_end(epoch, logs)
        for key in self.keys:
            self.s_logger.kpi[key] = float(logs[key])
        time_per_epoch = time.time() - self._epoch_start_time
        monitored_data = {
            "epoch": epoch,
            "time_per_epoch": str(timedelta(seconds=time_per_epoch)),
        }
        self.s_logger.write(
            data=monitored_data,
            status_level=status_logging.Status.RUNNING)
