# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
"""Utility class for performing TensorRT image inference."""

import numpy as np
import tensorrt as trt

from nvidia_tao_tf2.cv.classification.inferencer.inferencer import Inferencer
from nvidia_tao_tf2.cv.classification.inferencer.engine import allocate_buffers, do_inference, load_engine
# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
TRT_DYNAMIC_DIM = -1


class TRTInferencer(Inferencer):
    """Manages TensorRT objects for model inference."""

    def __init__(self, model_path, input_shape=None, batch_size=None,
                 img_mean=None, keep_aspect_ratio=False,
                 data_format='channel_first', img_depth=8):
        """Initializes TensorRT objects needed for model inference.

        Args:
            model_path (str): path where TensorRT engine should be stored
            input_shape (tuple): (batch, channel, height, width) for dynamic shape engine
            batch_size (int): batch size for dynamic shape engine
            img_depth (int): depth of images, only support 8-bit or 16-bit
            data_format (str): 'channel_first' or 'channel_last'
        """
        # We first load all custom plugins shipped with TensorRT,
        # some of them will be needed during inference
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = load_engine(self.trt_runtime, model_path)
        self.max_batch_size = batch_size or self.trt_engine.max_batch_size
        self.execute_v2 = True
        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

        def override_shape(shape, batch_size):
            return (batch_size if dim == TRT_DYNAMIC_DIM else dim for dim in shape)

        # Allocate memory for multiple usage [e.g. multiple batch inference]
        # Resolve dynamic shapes in the context
        for binding in self.trt_engine:
            binding_idx = self.trt_engine.get_binding_index(binding)
            shape = input_shape or self.trt_engine.get_binding_shape(binding_idx)

            if self.trt_engine.binding_is_input(binding_idx):
                assert binding_idx == 0, "More than 1 input is detected."
                if TRT_DYNAMIC_DIM in shape:
                    shape = override_shape(shape, self.max_batch_size)
                    self.execute_v2 = True
                self.context.set_binding_shape(binding_idx, shape)
                self._input_shape = shape
                if data_format == "channels_first":
                    self._img_height, self._img_width = self._input_shape[2:4]
                    self._nchannels = self._input_shape[1]
                else:
                    self._img_height, self._img_width = self._input_shape[1:3]
                    self._nchannels = self._input_shape[3]
                self.model_img_mode = 'RGB' if self._nchannels == 3 else 'L'

        assert self._input_shape, "Input shape not detected."
        assert self._nchannels in [1, 3], "Invalid input image dimension."

        print(f"TensorRT engine input shape: {self._input_shape}")
        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.trt_engine,
            self.context)

        input_volume = trt.volume(self._input_shape)
        self.numpy_array = np.zeros((self.max_batch_size, input_volume))
        self.img_mean = img_mean
        self.keep_aspect_ratio = keep_aspect_ratio
        self.model_img_mode = 'RGB' if self._nchannels == 3 else 'L'
        self.img_depth = img_depth
        assert self.img_depth in [8, 16], "Only 8-bit and 16-bit images are supported"

    def clear_buffers(self):
        """Simple function to free input, output buffers allocated earlier.

        Args:
            No explicit arguments. Inputs and outputs are member variables.
        Returns:
            No explicit returns.
        Raises:
            ValueError if buffers not found.
        """
        # Loop through inputs and free inputs.
        for inp in self.inputs:
            inp.device.free()

        # Loop through outputs and free them.
        for out in self.outputs:
            out.device.free()

    def clear_trt_session(self):
        """Simple function to free destroy tensorrt handlers.

        Args:
            No explicit arguments. Destroys context, runtime and engine.
        Returns:
            No explicit returns.
        Raises:
            ValueError if buffers not found.
        """
        if self.trt_runtime:
            del self.trt_runtime

        if self.context:
            del self.context

        if self.trt_engine:
            del self.trt_engine

        if self.stream:
            del self.stream

    def infer_single(self, img_path):
        """Infers model on batch of same sized images resized to fit the model.

        Args:
            img_path (str):  path to a single image file
        """
        # load image
        _, _, infer_input = self._load_img(img_path)
        infer_input = infer_input.transpose(2, 0, 1)
        imgs = infer_input[None, ...]
        # Verify if the supplied batch size is not too big
        max_batch_size = self.max_batch_size
        actual_batch_size = len(imgs)
        if actual_batch_size > max_batch_size:
            raise ValueError(
                f"image_paths list bigger ({actual_batch_size}) than engine max batch size ({max_batch_size})")
        self.numpy_array[:actual_batch_size] = imgs.reshape(actual_batch_size, -1)
        print(self.numpy_array.shape)
        # ...copy them into appropriate place into memory...
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, self.numpy_array.ravel())

        # ...fetch model outputs...
        results = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream,
            batch_size=max_batch_size,
            execute_v2=self.execute_v2)

        # ...and return results up to the actual batch size.
        return results

    def __del__(self):
        """Clear things up on object deletion."""
        self.clear_trt_session()
        self.clear_buffers()
