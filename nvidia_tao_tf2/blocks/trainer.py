# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""TAO Trainer base class."""

from abc import ABC, abstractmethod


class Trainer(ABC):
    """Abstract class for TAO model trainer.

    Usage: TODO
    """

    @abstractmethod
    def __init__(self, model):
        """Initialize."""
        pass

    def set_callbacks(self, callbacks):
        """Set callbacks."""
        self.callbacks = callbacks

    @abstractmethod
    def fit(self, train_dataset, eval_dataset,
            config=None):
        """Run model fitting."""
        # 1. override config
        # 2. check config is dict
        # 3. call model.fit or custom loop
        pass

    def train_step(self, step, x, y):
        """Train step."""
        pass

    def test_step(self, step, x, y):
        """Test step."""
        pass
