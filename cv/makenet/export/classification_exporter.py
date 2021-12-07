# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.

"""Base class to export trained .tlt models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tf2onnx
import tensorrt as trt
import tensorflow as tf

from model_optimization.quantization.qdq_layer import QDQ
from model_optimization.quantization.quantized_conv2d import QuantizedConv2D

TRT_VERSION = trt.__version__
TRT_MAJOR = float(".".join(TRT_VERSION.split(".")[:2]))
logger = logging.getLogger(__name__)


class Exporter:
    """Define an exporter for classification models."""

    def __init__(self,
                 config=None,
                 dtype="fp32",
                 strict_type=False,
                 classmap_file=None,
                 max_batch_size=1,
                 min_batch_size=1,
                 opt_batch_size=1,
                 **kwargs):
        """Initialize the classification exporter.

        Args:
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
        """
        self.config = config
        if dtype == "int8":
            self._dtype = trt.DataType.INT8
        elif dtype == "fp16":
            self._dtype = trt.DataType.HALF
        elif dtype == "fp32":
            self._dtype = trt.DataType.FLOAT
        else:
            raise ValueError("Unsupported data type: %s" % dtype)
        self.strict_type = strict_type
        self.backend = "onnx"
        self.classmap_file = classmap_file
        self.model = None
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.opt_batch_size = opt_batch_size
        self.input_shape = None

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
        self._set_input_shape()
    
    def _set_input_shape(self):
        self.input_shape = tuple(self.model.layers[0].input_shape[0][1:4])

    def export_onnx(self, onnx_path) -> None:
        """Export to ONNX"""
        spec = (tf.TensorSpec((None, ) + self.input_shape, tf.float32, name="input_1"),)
        model_proto, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=13, output_path=onnx_path)
        output_names = [n.name for n in model_proto.graph.output]
        logger.info(f"output names: {output_names}")
    
    def export_engine(self, onnx_path, engine_path, save_engine=True, verbose=True) -> None:
        """Parse the model file through TensorRT and build TRT engine
        """
        # Create builder and network
        if verbose:
            TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        else:
            TRT_LOGGER = trt.Logger(trt.Logger.INFO)

        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network_flags = network_flags | (
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)
        )

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                flags=network_flags
        ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(onnx_path, "rb") as model:
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
                else:
                    print("Parsed ONNX model successfully")

            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30
            builder.max_batch_size = self.max_batch_size
            if TRT_MAJOR >= 8.2:
                config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            else:
                config.profiling_verbosity = trt.ProfilingVerbosity.VERBOSE

            if self._dtype == trt.DataType.HALF:
                config.set_flag(trt.BuilderFlag.FP16)

            if self._dtype == trt.DataType.INT8:
                config.set_flag(trt.BuilderFlag.INT8)
            # config.flags = config.flags | 1 << int(trt.BuilderFlag.INT8)
            # Setting the (min, opt, max) batch sizes to be (1, 4, 8).
            #   The users need to configure this according to their requirements.
            config.add_optimization_profile(
                self._build_profile(
                    builder,
                    network,
                    profile_shapes={
                        "input_1": [(self.min_batch_size,) + self.input_shape,
                                (self.opt_batch_size,) + self.input_shape,
                                (self.max_batch_size,) + self.input_shape]
                    },
                )
            )
            trt_engine = builder.build_engine(network, config)  # build_serialized_network
            if not trt_engine:
                logger.info("TensorRT engine failed.")
            if save_engine:
                with open(engine_path, "wb") as engine_file:
                    engine_file.write(trt_engine.serialize())

        
    def _build_profile(self, builder, network, profile_shapes, default_shape_value=1):
        """
        Build optimization profile for the builder and configure the min, opt, max shapes appropriately.
        """

        def is_dimension_dynamic(dim):
            return dim is None or dim <= 0

        def override_shape(shape):
            return tuple([1 if is_dimension_dynamic(dim) else dim for dim in shape])

        profile = builder.create_optimization_profile()
        for idx in range(network.num_inputs):
            inp = network.get_input(idx)

            def get_profile_shape(name):
                # if name not in profile_shapes:
                # profile_shapes={'input': [(),(),()]} and name='input_1:0
                profile_name = None
                for k in profile_shapes.keys():
                    if k in name:
                        profile_name = k
                if profile_name is None:  # not any([k in name for k in profile_shapes.keys()]):
                    return None
                shapes = profile_shapes[profile_name]
                if not isinstance(shapes, list) or len(shapes) != 3:
                    logger.critical(
                        "Profile values must be a list containing exactly 3 shapes (tuples or Dims), but received shapes: {:} for input: {:}.\nNote: profile was: {:}.\nNote: Network inputs were: {:}".format(
                            shapes, name, profile_shapes, shapes,
                        )
                    )
                return shapes

            if inp.is_shape_tensor:
                shapes = get_profile_shape(inp.name)
                if not shapes:
                    rank = inp.shape[0]
                    shapes = [(default_shape_value,) * rank] * 3
                    print(
                        "Setting shape input to {:}. If this is incorrect, for shape input: {:}, please provide tuples for min, opt, and max shapes containing {:} elements".format(
                            shapes[0], inp.name, rank
                        )
                    )
                min, opt, max = shapes
                profile.set_shape_input(inp.name, min, opt, max)
                print(
                    "Setting shape input: {:} values to min: {:}, opt: {:}, max: {:}".format(
                        inp.name, min, opt, max
                    )
                )
            elif -1 in inp.shape:
                shapes = get_profile_shape(inp.name)
                if not shapes:
                    shapes = [override_shape(inp.shape)] * 3
                    print(
                        "Overriding dynamic input shape {:} to {:}. If this is incorrect, for input tensor: {:}, please provide tuples for min, opt, and max shapes containing values: {:} with dynamic dimensions replaced,".format(
                            inp.shape, shapes[0], inp.name, inp.shape
                        )
                    )
                min, opt, max = shapes
                profile.set_shape(inp.name, min, opt, max)
                print(
                    "Setting input: {:} shape to min: {:}, opt: {:}, max: {:}".format(
                        inp.name, min, opt, max
                    )
                )
        if not profile:
            print(
                "Profile is not valid, please provide profile data. Note: profile was: {:}".format(
                    profile_shapes
                )
            )
        return profile
