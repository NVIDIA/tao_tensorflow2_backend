# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

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

    @abstractmethod
    def train_step(self, data):
        """Train step."""
        pass

    @abstractmethod
    def test_step(self, data):
        """Test step."""
        pass
