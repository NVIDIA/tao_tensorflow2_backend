# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.

"""TAO Trainer base class."""

from abc import ABC, abstractmethod


class Trainer(ABC):
    """Abstract class for TAO model trainer.
    
    Usage: TODO
    """
    @abstractmethod
    def __init__(self, model):
        """init function."""
        pass
    
    def set_callbacks(self, callbacks):
        """Set callbacks."""
        self.callbacks = callbacks

    @abstractmethod
    def fit(self, train_dataset, eval_dataset,
            config=None):
        # 1. override config
        # 2. check config is dict
        # 3. call model.fit or custom loop
        pass

    def train_step(self, step, x, y):
        pass

    def test_step(self, step, x, y):
        pass
    