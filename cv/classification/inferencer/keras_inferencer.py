# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Utility class for performing TensorRT image inference."""

import numpy as np

from cv.classification.inferencer.inferencer import Inferencer
from cv.classification.utils.helper import load_model
from cv.classification.utils.preprocess_crop import load_and_crop_img
from cv.classification.utils.preprocess_input import preprocess_input


class KerasInferencer(Inferencer):
    """Keras inferencer"""

    def __init__(self, model_path,
                 img_mean=None,
                 keep_aspect_ratio=True,
                 key=None,
                 preprocess_mode='torch',
                 interpolation='bilinear',
                 img_depth=8):
        """Init."""
        self.model_path = model_path
        self.img_mean = img_mean
        self.keep_aspect_ratio = keep_aspect_ratio
        self.key = key
        self.preprocess_mode = preprocess_mode
        self.interpolation = interpolation
        self.img_depth = img_depth
        self._load_model(model_path)

    def _load_model(self, model_path) -> None:

        self.model = load_model(model_path, self.key)
        self.model.summary()

        self._input_shape = tuple(self.model.layers[0].input_shape[0])
        self._data_format = self.model.layers[1].data_format
        if self._data_format == "channels_first":
            self._img_height, self._img_width = self._input_shape[2:4]
            self._nchannels = self._input_shape[1]
        else:
            self._img_height, self._img_width = self._input_shape[1:3]
            self._nchannels = self._input_shape[3]
        self.model_img_mode = 'rgb' if self._nchannels == 3 else 'grayscale'

    def _load_img(self, img_path):
        image = load_and_crop_img(
            img_path,
            grayscale=False,
            color_mode=self.model_img_mode,
            target_size=(self._img_height, self._img_width),
            interpolation=self.interpolation,
        )
        image = np.array(image).astype(np.float32)
        return preprocess_input(
            image,
            mode=self.preprocess_mode, color_mode=self.model_img_mode,
            img_mean=self.img_mean,
            data_format='channels_last')

    def infer_single(self, img_path):
        """Run inference on a single image with the tlt model."""
        infer_input = self._load_img(img_path)
        if self._data_format == "channels_first":
            infer_input = infer_input.transpose(2, 0, 1)
        infer_input = infer_input[None, ...]
        # Keras inference
        raw_predictions = self.model.predict(infer_input, batch_size=1)
        return raw_predictions
