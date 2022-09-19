# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EfficientDet TensorRT engine builder."""

import logging
import os
import sys

import numpy as np
import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt

from cv.efficientdet.exporter.image_batcher import ImageBatcher

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


class EngineCalibrator(trt.IInt8EntropyCalibrator2):
    """Implements the INT8 Entropy Calibrator2."""

    def __init__(self, cache_file):
        """Init.

        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """Define the image batcher to use, if any.

        If using only the cache file,
        an image batcher doesn't need to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        size = int(np.dtype(self.image_batcher.dtype).itemsize * np.prod(self.image_batcher.shape))
        self.batch_allocation = cuda.mem_alloc(size)
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch, _, _ = next(self.batch_generator)
            log.info("Calibrating image {} / {}".format(   # noqa pylint: disable=C0209
                self.image_batcher.image_index, self.image_batcher.num_images))
            cuda.memcpy_htod(self.batch_allocation, np.ascontiguousarray(batch))
            return [int(self.batch_allocation)]
        except StopIteration:
            log.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                log.info("Using calibration cache file: {}".format(self.cache_file))   # noqa pylint: disable=C0209
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        """Overrides from trt.IInt8EntropyCalibrator2.

        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        with open(self.cache_file, "wb") as f:
            log.info("Writing calibration cache data to: {}".format(self.cache_file))  # noqa pylint: disable=C0209
            f.write(cache)


class EngineBuilder:
    """Parses an ONNX graph and builds a TensorRT engine from it."""

    def __init__(self, verbose=False, workspace=8, is_qat=False):
        """Init.

        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        :param workspace: Max memory workspace to allow, in Gb.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * (2 ** 30)

        # self.batch_size = None
        self.network = None
        self.parser = None
        self.is_qat = is_qat

    def create_network(self, onnx_path, batch_size, dynamic_batch_size=None):
        """Parse the ONNX graph and create the corresponding TensorRT network definition.

        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))  # noqa pylint: disable=C0209
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        log.info("Network Description")
        profile = self.builder.create_optimization_profile()
        dynamic_inputs = False
        for inp in inputs:
            log.info("Input '{}' with shape {} and dtype {}".format(inp.name, inp.shape, inp.dtype))  # noqa pylint: disable=C0209
            if inp.shape[0] == -1:
                dynamic_inputs = True
                if dynamic_batch_size:
                    if type(dynamic_batch_size) is str:
                        dynamic_batch_size = [int(v) for v in dynamic_batch_size.split(",")]
                    assert len(dynamic_batch_size) == 3
                    min_shape = [dynamic_batch_size[0]] + list(inp.shape[1:])
                    opt_shape = [dynamic_batch_size[1]] + list(inp.shape[1:])
                    max_shape = [dynamic_batch_size[2]] + list(inp.shape[1:])
                    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                    log.info("Input '{}' Optimization Profile with shape MIN {} / OPT {} / MAX {}".format(  # noqa pylint: disable=C0209
                        inp.name, min_shape, opt_shape, max_shape))
                else:
                    shape = [batch_size] + list(inp.shape[1:])
                    profile.set_shape(inp.name, shape, shape, shape)
                    log.info("Input '{}' Optimization Profile with shape {}".format(inp.name, shape))  # noqa pylint: disable=C0209
        if dynamic_inputs:
            self.config.add_optimization_profile(profile)

    def create_engine(self, engine_path, precision,
                      calib_input=None, calib_cache=None, calib_num_images=5000,
                      calib_batch_size=8):
        """Build the TensorRT engine and serialize it to disk.

        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        :param calib_input: The path to a directory holding the calibration images.
        :param calib_cache: The path where to write the calibration cache to,
        or if it already exists, load it from.
        :param calib_num_images: The maximum number of images to use for calibration.
        :param calib_batch_size: The batch size to use for the calibration process.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.debug("Building {} Engine in {}".format(precision, engine_path))  # noqa pylint: disable=C0209

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8":
            if not self.builder.platform_has_fast_int8:
                log.warning("INT8 is not supported natively on this platform/device")
            elif self.is_qat:
                print("Exporting a QAT model...")
                self.config.set_flag(trt.BuilderFlag.INT8)
            else:
                assert calib_cache, "cal_cache_file must be specified when exporting a model in PTQ INT8 mode."
                if self.builder.platform_has_fast_fp16:
                    # Also enable fp16, as some layers may be even more efficient in fp16 than int8
                    self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.INT8)
                self.config.int8_calibrator = EngineCalibrator(calib_cache)
                if not os.path.exists(calib_cache):
                    calib_shape = [calib_batch_size] + list(inputs[0].shape[1:])
                    calib_dtype = trt.nptype(inputs[0].dtype)
                    self.config.int8_calibrator.set_image_batcher(
                        ImageBatcher(calib_input, calib_shape, calib_dtype,
                                     max_num_images=calib_num_images,
                                     exact_batches=True))

        engine_bytes = None
        try:
            engine_bytes = self.builder.build_serialized_network(self.network, self.config)
        except AttributeError:
            engine = self.builder.build_engine(self.network, self.config)
            engine_bytes = engine.serialize()
            del engine
        assert engine_bytes
        with open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))  # noqa pylint: disable=C0209
            f.write(engine_bytes)
