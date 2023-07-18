"""Data preprocessing."""

from abc import ABC, abstractmethod


class Postprocessor(ABC):
    """Base class of Input processor."""

    @abstractmethod
    def __init__(self, config=None):
        """Initializes a new `Postprocessor`.

        Args:
        config: postprocessing config
        """
        pass

    @abstractmethod
    def generate_detections(self, model_outputs):
        """Process model outputs with postprocessing ops."""
        pass
