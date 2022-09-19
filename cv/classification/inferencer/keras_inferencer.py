# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Utility class for performing TensorRT image inference."""

import numpy as np
from PIL import Image
import tensorflow as tf

from cv.classification.inferencer.inferencer import Inferencer
from cv.classification.utils.helper import decode_eff
from model_optimization.quantization.qdq_layer import QDQ
from model_optimization.quantization.quantized_conv2d import QuantizedConv2D

class KerasInferencer(Inferencer):
    """Keras inferencer"""
    def __init__(self, model_path,
                 img_mean=None,
                 keep_aspect_ratio=False,
                 key=None,
                 img_depth=8):

        self.model_path = model_path
        self.img_mean = img_mean
        self.keep_aspect_ratio = keep_aspect_ratio
        self.key = key
        self.img_depth = img_depth
        self._load_model(model_path)
        
    def _load_model(self, model_path) -> None:
        custom_objs = {
            'QDQ': QDQ,
            'QuantizedConv2D': QuantizedConv2D,
        }
        if self.key:
            model_path = decode_eff(model_path, self.key)
        self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
        self.model.summary()

        self._input_shape = tuple(self.model.layers[0].input_shape[0])
        self._data_format = self.model.layers[1].data_format
        if self._data_format == "channels_first":
            self._img_height, self._img_width = self._input_shape[2:4] 
            self._nchannels =  self._input_shape[1] 
        else:
            self._img_height, self._img_width = self._input_shape[1:3]
            self._nchannels = self._input_shape[3] 
        self.model_img_mode = 'RGB' if self._nchannels == 3 else 'L'

    def infer_single(self, img_path):
        _, _, infer_input = self._load_img(img_path)
        if self._data_format == "channels_first":
            infer_input = infer_input.transpose(2, 0, 1)
        infer_input = infer_input[None, ...]
        # Keras inference
        raw_predictions = self.model.predict(infer_input, batch_size=1)
        return raw_predictions
