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

"""TAO Trainer base class."""

from abc import ABC, abstractmethod


class Trainer(ABC):
    """Abstract class for TAO model trainer."""

    def __init__(self, model, config, callbacks=None):
        """Initialize."""
        self.model = model
        self.callbacks = callbacks
        self.config = config

    def set_callbacks(self, callbacks):
        """Set callbacks."""
        self.callbacks = callbacks

    @abstractmethod
    def fit(self, train_dataset, eval_dataset,
            num_epochs,
            steps_per_epoch,
            initial_epoch,
            validation_steps,
            verbose):
        """Run model fitting."""
        pass

    def train_step(self, data):
        """Train step."""
        pass

    def test_step(self, data):
        """Test step."""
        pass
