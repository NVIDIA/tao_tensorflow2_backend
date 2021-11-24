# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Base class to export trained .tlt models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tf2onnx
import tensorflow as tf

from quantization.qdq_layer import QDQ
from quantization.quantized_conv2d import QuantizedConv2D

logger = logging.getLogger(__name__)


class Exporter:
    """Define an exporter for classification models."""

    def __init__(self,
                 config=None,
                 data_type="fp32",
                 strict_type=False,
                 classmap_file=None,
                 **kwargs):
        """Initialize the classification exporter.

        Args:
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
        """
        self.config = config
        self.data_type = data_type
        self.strict_type = strict_type
        self.backend = "onnx"
        self.classmap_file = classmap_file
        self.model = None

    def load_model(self, model_path, key=None) -> None:
        custom_objs = {
            'QDQ': QDQ,
            'QuantizedConv2D': QuantizedConv2D,
        }
        if not key:
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objs)
            self.model.summary()
        else:
            raise NotImplementedError
        
    def export(self, out_path):
        spec = (tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input_1"),)
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13, output_path=out_path)
        output_names = [n.name for n in model_proto.graph.output]
        logger.info(f"output names: {output_names}")
        