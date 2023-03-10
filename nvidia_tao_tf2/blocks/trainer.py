# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

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
