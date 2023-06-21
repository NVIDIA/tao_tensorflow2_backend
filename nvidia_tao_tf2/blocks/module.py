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

"""TAO Module base class."""

from abc import ABC, abstractmethod


class TAOModule(ABC):
    """Abstract class for TAO Module."""

    @abstractmethod
    def __init__(self, hparams) -> None:
        """Init."""
        pass

    @abstractmethod
    def configure_losses(self):
        """Configure losses."""
        pass

    @abstractmethod
    def configure_optimizers(self):
        """Configure optimizers."""
        pass

    @abstractmethod
    def compile(self):
        """Compile model."""
        pass

    def train_step(self, data):
        """Train step."""
        pass

    def test_step(self, data):
        """Test step."""
        pass
