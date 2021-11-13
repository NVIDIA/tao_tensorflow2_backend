"""Data preprocessing."""

from abc import ABC, abstractmethod


class Preprocessor(ABC):
    """Base class of Input processor."""
    @abstractmethod
    def __init__(self, images, output_size):
        """Initializes a new `InputProcessor`.

        Args:
        images: The input images before processing.
        output_size: The output image size after calling resize_and_crop_image
            function.
        """
        pass

    @abstractmethod
    def transform(self):
        """Process input images with a series of preprocessing ops."""
        pass