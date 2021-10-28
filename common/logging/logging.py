# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Logger class for TLT IVA models."""

from abc import abstractmethod
from datetime import datetime
import json
import logging
import os

logger = logging.getLogger(__name__)


class BaseLogger(object):
    """File logger class."""

    def __init__(self, is_master=False):
        """Base logger class."""
        self.is_master = is_master

    @abstractmethod
    def write(self, data=None, status_string=None):
        """Write data out to the log file."""
        logging.info(status_string)
        if data:
            if not isinstance(data, dict):
                raise NotImplementedError("The provided data should be a dictionary")
            data_string = json.dumps(data, indent=4)
            logging.info(data_string)


class StatusLogger(BaseLogger):
    """Simple logger to save the status file."""

    def __init__(self, filename=None, is_master=False):
        """Logger to write out the status."""
        super().__init__(is_master=is_master)
        self.log_path = os.path.realpath(filename)
        if os.path.exists(self.log_path):
            logger.info("Log file already exists at {}".format(self.log_path))

    @property
    def date(self):
        """Get date from the status."""
        date_time = datetime.now()
        date_object = date_time.date()
        return "{}/{}/{}".format(
            date_object.month,
            date_object.day,
            date_object.year
        )

    @property
    def time(self):
        """Get date from the status."""
        date_time = datetime.now()
        time_object = date_time.time()
        return "{}:{}:{}".format(
            time_object.hour,
            time_object.minute,
            time_object.second
        )

    def write(self, data=None, status_string=None):
        """Function to write data to the status json file."""
        if not data:
            data = {}

        data["date"] = self.date
        data["time"] = self.time
        if status_string:
            data["status"] = status_string

        # Write out status only from rank 0.
        if self.is_master:
            with open(self.log_path, "a") as logfile:
                logfile.write("{}\n".format(json.dumps(data)))


# Define the logger here so it's static.
_STATUS_LOGGER = BaseLogger()


def set_status_logger(status_logger):
    """Set the status logger.

    Args:
        status_logger: An instance of the logger class.
    """
    global _STATUS_LOGGER  # pylint: disable=W0603
    _STATUS_LOGGER = status_logger


def get_status_logger():
    """Get the status logger."""
    global _STATUS_LOGGER  # pylint: disable=W0602,W0603
    return _STATUS_LOGGER
