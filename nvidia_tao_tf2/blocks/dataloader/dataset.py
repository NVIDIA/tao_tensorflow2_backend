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
