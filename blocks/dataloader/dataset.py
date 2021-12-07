# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""TAO Dataset base class."""

from abc import ABC, abstractmethod


class Dataset(ABC):
    """Abstract class for TAO TF Dataset."""

    @abstractmethod
    def __init__(self, file_pattern, is_training,
                 use_fake_data=False, max_instances_per_image=None):
        """init function."""
        pass

    @abstractmethod
    def dataset_parser(self, value, example_decoder, configs=None):
        """Parse data to a fixed dimension input image and learning targets.

        Args:
            value: a single serialized tf.Example string.
            example_decoder: TF example decoder.
        """
        pass

    @abstractmethod
    def process_example(self, values, config=None):
        """Processes one batch of data."""
        pass

    