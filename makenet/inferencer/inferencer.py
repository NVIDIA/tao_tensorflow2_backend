# Copyright 2019-2020 NVIDIA Corporation.  All rights reserved.
"""Utility class for performing TensorRT image inference."""

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from makenet.utils.preprocess_input import preprocess_input


class Inferencer(ABC):
    """Manages model inference."""
    @abstractmethod
    def __init__(self, model_path, input_shape=None, batch_size=None,
                 img_mean=None, keep_aspect_ratio=False):
        pass
    
    @abstractmethod
    def infer_single(self, img_path):
        pass

    def _load_img(self, img_path=None, img_alt=None):
        """load an image and returns the original image and a numpy array for model to consume.

        Args:
            img_path (str): path to an image
            img_alt (np.array): only for testing (h, w, c)
        Returns:
            img (PIL.Image): PIL image of original image.
            ratio (float): resize ratio of original image over processed image
            inference_input (array): numpy array for processed image
        """
        if img_path:
            img = Image.open(img_path)
        elif img_alt is not None:
            img = Image.fromarray(img_alt)
        else:
            raise RuntimeError("image path is not defined.")
        orig_w, orig_h = img.size
        ratio = min(self._input_shape[-1]/float(orig_w), self._input_shape[-2]/float(orig_h))

        # do not change aspect ratio
        new_w = int(round(orig_w*ratio))
        new_h = int(round(orig_h*ratio))

        if self.keep_aspect_ratio:
            im = img.resize((new_w, new_h), Image.ANTIALIAS)
        else:
            im = img.resize((self._input_shape[-1], self._input_shape[-2]), Image.ANTIALIAS)

        if im.mode in ('RGBA', 'LA') or \
                (im.mode == 'P' and 'transparency' in im.info) and \
                self.model_img_mode == 'L':

            # Need to convert to RGBA if LA format due to a bug in PIL
            im = im.convert('RGBA')
            inf_img = Image.new("RGBA", (self._input_shape[-1], self._input_shape[-2]))
            inf_img.paste(im, (0, 0))
            inf_img = inf_img.convert(self.model_img_mode)
        else:
            inf_img = Image.new(
                self.model_img_mode,
                (self._input_shape[-1], self._input_shape[-2])
            )
            inf_img.paste(im, (0, 0))

        inf_img = np.array(inf_img).astype(np.float32)
        if self.model_img_mode == 'L':
            inf_img = np.expand_dims(inf_img, axis=2)
            inference_input = inf_img - 117.3786
        else:
            inference_input = preprocess_input(inf_img,
                                               img_mean=self.img_mean)

        return img, float(orig_w)/new_w, inference_input