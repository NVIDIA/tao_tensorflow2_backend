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

"""Base class to export trained .tlt models to etlt file."""

import copy
import os
import shutil
import logging
import tempfile

import onnx
import onnx_graphsurgeon as gs

import tensorrt as trt
import tensorflow as tf

from tf2onnx import tf_loader, convert

from nvidia_tao_tf2.cv.classification.utils.helper import decode_eff

TRT_VERSION = trt.__version__
TRT_MAJOR = float(".".join(TRT_VERSION.split(".")[:2]))
logger = logging.getLogger(__name__)


class Exporter:
    """Define an exporter for classification models."""

    def __init__(self,
                 config=None,
                 min_batch_size=1,
                 opt_batch_size=4,
                 max_batch_size=8,
                 **kwargs):
        """Initialize the classification exporter."""
        self.config = config
        self.backend = "onnx"
        self.input_shape = None
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.opt_batch_size = opt_batch_size

        self._saved_model = decode_eff(
            str(self.config.export.checkpoint),
            self.config.encryption_key)
        _handle, self.tmp_onnx = tempfile.mkstemp(suffix='onnx')
        os.close(_handle)

    def _set_input_shape(self):
        model = tf.keras.models.load_model(self._saved_model, custom_objects=None)
        self.input_shape = tuple(model.layers[0].input_shape[0][1:4])

    def export_onnx(self) -> None:
        """Convert Keras saved model into ONNX format.

        Args:
            root_dir: root directory containing the quantized Keras saved model. This is the same directory where the ONNX
                file will be saved.
            saved_model_dir: name of the quantized 'saved_model' directory.
            onnx_filename: desired name to save the converted ONNX file.
        """
        # 1. Let TensorRT optimize QDQ nodes instead of TF
        from tf2onnx.optimizer import _optimizers  # noqa pylint: disable=C0415
        updated_optimizers = copy.deepcopy(_optimizers)
        del updated_optimizers["q_dq_optimizer"]
        del updated_optimizers["const_dequantize_optimizer"]

        # 2. Extract graph definition from SavedModel
        graph_def, inputs, outputs = tf_loader.from_saved_model(
            model_path=self._saved_model,
            input_names=None,
            output_names=None,
            tag="serve",
            signatures=["serving_default"]
        )

        # 3. Convert tf2onnx and save onnx file
        if str(self.config.export.onnx_file).endswith('.onnx'):
            onnx_path = self.config.export.onnx_file
        else:
            raise ValueError("The exported file must use .onnx as the extension.")
        model_proto, _ = convert._convert_common(
            graph_def,
            name=self._saved_model,
            opset=13,
            input_names=inputs,
            output_names=outputs,
            output_path=onnx_path,
            optimizers=updated_optimizers
        )
        graph = gs.import_onnx(model_proto)
        graph.inputs[0].name = "input_1"
        graph.cleanup().toposort()
        onnx_model = gs.export_onnx(graph)
        onnx.save(onnx_model, onnx_path)
        logger.info("ONNX conversion completed.")

    def _del(self):
        """Remove temp files."""
        shutil.rmtree(self._saved_model)

    def export(self):
        """Export ONNX model."""
        self._set_input_shape()
        self.export_onnx()
        logger.info("The ONNX model is saved at %s", self.config.export.onnx_file)
        self._del()

    def _build_profile(self, builder, network, profile_shapes, default_shape_value=1):
        """Build optimization profile for the builder and configure the min, opt, max shapes appropriately."""
        def is_dimension_dynamic(dim):
            return dim is None or dim <= 0

        def override_shape(shape):
            return (1 if is_dimension_dynamic(dim) else dim for dim in shape)

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
                    logger.critical("Profile values must be a list containing exactly 3 shapes (tuples or Dims)")
                return shapes

            if inp.is_shape_tensor:
                shapes = get_profile_shape(inp.name)
                if not shapes:
                    rank = inp.shape[0]
                    shapes = [(default_shape_value,) * rank] * 3
                    print(
                        "Setting shape input to {:}. If this is incorrect, for shape input: {:}, please provide tuples for min, opt, and max shapes containing {:} elements".format(  # noqa pylint: disable=C0209
                            shapes[0], inp.name, rank
                        )
                    )
                min, opt, max = shapes   # noqa pylint: disable=W0622
                profile.set_shape_input(inp.name, min, opt, max)
                print(
                    "Setting shape input: {:} values to min: {:}, opt: {:}, max: {:}".format(  # noqa pylint: disable=C0209
                        inp.name, min, opt, max
                    )
                )
            elif -1 in inp.shape:
                shapes = get_profile_shape(inp.name)
                if not shapes:
                    shapes = [override_shape(inp.shape)] * 3
                    print(
                        "Overriding dynamic input shape {:} to {:}. If this is incorrect, for input tensor: {:}, please provide tuples for min, opt, and max shapes containing values: {:} with dynamic dimensions replaced,".format(  # noqa pylint: disable=C0209
                            inp.shape, shapes[0], inp.name, inp.shape
                        )
                    )
                min, opt, max = shapes
                profile.set_shape(inp.name, min, opt, max)
                print(
                    "Setting input: {:} shape to min: {:}, opt: {:}, max: {:}".format(  # noqa pylint: disable=C0209
                        inp.name, min, opt, max
                    )
                )
        if not profile:
            print(
                "Profile is not valid, please provide profile data. Note: profile was: {:}".format(  # noqa pylint: disable=C0209
                    profile_shapes
                )
            )
        return profile
