# Copyright (c) 2017-2020, NVIDIA CORPORATION.  All rights reserved.

"""Base class to export trained .tlt keras models to etlt file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import struct
import tempfile

import keras
from magnet import encoding

# Import quantization layer processing.
from modulus.export._quantized import (
    check_for_quantized_layers,
    process_quantized_layers,
)
from modulus.export._tensorrt import ONNXEngineBuilder, UFFEngineBuilder
from modulus.export._uff import keras_to_uff
from modulus.export.app import get_model_input_dtype

import tensorflow as tf

from iva.common.export.base_exporter import BaseExporter
from iva.common.types.base_ds_config import BaseDSConfig
from iva.common.utils import (
    CUSTOM_OBJS,
    get_decoded_filename,
    get_model_file_size,
    get_num_params,
    model_io
)

VALID_BACKEND = ["uff", "onnx"]

logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class KerasExporter(BaseExporter):
    """Base class for exporter."""

    def __init__(self, model_path=None,
                 key=None,
                 data_type="fp32",
                 strict_type=False,
                 backend="uff",
                 data_format="channels_first",
                 **kwargs):
        """Initialize the base exporter.

        Args:
            model_path (str): Path to the model file.
            key (str): Key to load the model.
            data_type (str): Path to the TensorRT backend data type.
            strict_type(bool): Apply TensorRT strict_type_constraints or not for INT8 mode.
            backend (str): TensorRT parser to be used.

        Returns:
            None.
        """
        super(KerasExporter, self).__init__(
              model_path=model_path,
              data_type=data_type,
              strict_type=strict_type,
              key=key,
              backend=backend,
              **kwargs
        )
        self.data_format = data_format
        self._onnx_graph_node_name_dict = None
        self._onnx_graph_node_op_dict = None
        self.image_mean = None
        self.experiment_spec = None
        self.model_param_count = None

    def set_session(self):
        """Set keras backend session."""
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        keras.backend.set_session(tf.Session(config=config))

    def set_keras_backend_dtype(self):
        """Set the keras backend data type."""
        keras.backend.set_learning_phase(0)
        tmp_keras_file_name = get_decoded_filename(self.model_path,
                                                   self.key)
        model_input_dtype = get_model_input_dtype(tmp_keras_file_name)
        keras.backend.set_floatx(model_input_dtype)

    def set_input_output_node_names(self):
        """Set input output node names."""
        raise NotImplementedError("This function is not implemented in the base class.")

    @staticmethod
    def extract_tensor_scale(model, backend):
        """Extract tensor scale from QAT trained model and de-quantize the model."""
        model, tensor_scale_dict = process_quantized_layers(
            model, backend,
            calib_cache=None,
            calib_json=None)
        return model, tensor_scale_dict

    def load_model(self, backend="uff"):
        """Simple function to get the keras model."""
        keras.backend.clear_session()
        keras.backend.set_learning_phase(0)
        model = model_io(self.model_path, enc_key=self.key)
        if check_for_quantized_layers(model):
            model, self.tensor_scale_dict = self.extract_tensor_scale(model, backend)
        return model

    def set_backend(self, backend):
        """Set keras backend.

        Args:
            backend (str): Backend to be used.
                Currently only UFF is supported.
        """
        if backend not in VALID_BACKEND:
            raise NotImplementedError('Invalid backend "{}" called'.format(backend))
        self.backend = backend

    def _get_onnx_node_by_name(self, onnx_graph, node_name, clean_cache=False):
        '''Find onnx graph nodes by node_name.

        Args:
            onnx_graph: Input onnx graph
            node_name: The node name to find
            clean_cache: Whether to rebuild the graph node cache

        Returns:
            onnx_node

        It's your responsibility to make sure:
            1. The node names in the graph are unique and node_name exists in graph
            2. If clean_cache is False, the node name cache you build last time should work
        '''
        if self._onnx_graph_node_name_dict is None or clean_cache:
            self._onnx_graph_node_name_dict = {n.name: n for n in onnx_graph.nodes}

        return self._onnx_graph_node_name_dict[node_name]

    def _get_onnx_node_by_op(self, onnx_graph, op_name, clean_cache=False):
        '''Find onnx graph nodes by op_name.

        Args:
            onnx_graph: Input onnx graph
            op_name: The node op to find
            clean_cache: Whether to rebuild the graph node cache

        Returns:
            List of onnx_nodes

        It's your responsibility to make sure:
            1. If clean_cache is False, the node name cache you build last time should work
        '''
        if self._onnx_graph_node_op_dict is None or clean_cache:
            self._onnx_graph_node_op_dict = dict()
            for n in onnx_graph.nodes:
                if n.op in self._onnx_graph_node_op_dict:
                    self._onnx_graph_node_op_dict[n.op].append(n)
                else:
                    self._onnx_graph_node_op_dict[n.op] = [n]

        if op_name in self._onnx_graph_node_op_dict:
            return self._onnx_graph_node_op_dict[op_name]
        return []

    def _fix_onnx_paddings(self, graph):
        """Fix the paddings in onnx graph so it aligns with the Keras patch."""
        # third_party/keras/tensorflow_backend.py patched the semantics of
        # SAME padding, the onnx model has to align with it.
        for node in graph.nodes:
            if node.op == "Conv":
                # in case of VALID padding, there is no 'pads' attribute
                # simply skip it
                if node.attrs["auto_pad"] == "VALID":
                    continue
                k = node.attrs['kernel_shape']
                g = node.attrs['group']
                d = node.attrs['dilations']
                # always assume kernel shape is square
                effective_k = [1 + (k[ki] - 1) * d[ki] for ki in range(len(d))]
                # (pad_w // 2 , pad_h // 2) == (pad_left, pad_top)
                keras_paddings = tuple((ek - 1) // 2 for ek in effective_k)
                # (pad_left, pad_top, pad_right, pad_bottom)
                if g == 1:
                    # if it is not VALID, then it has to be NOTSET,
                    # to enable explicit paddings below
                    node.attrs["auto_pad"] = "NOTSET"
                    # only apply this patch for non-group convolutions
                    node.attrs['pads'] = keras_paddings * 2
            elif node.op in ["AveragePool", "MaxPool"]:
                # skip VALID padding case.
                if node.attrs["auto_pad"] == "VALID":
                    continue
                k = node.attrs['kernel_shape']
                # (pad_w // 2 , pad_h // 2) == (pad_left, pad_top)
                keras_paddings = tuple((ek - 1) // 2 for ek in k)
                # force it to be NOTSET to enable explicit paddings below
                node.attrs["auto_pad"] = "NOTSET"
                # (pad_left, pad_top, pad_right, pad_bottom)
                node.attrs['pads'] = keras_paddings * 2

    def save_etlt_file(self, model, output_file_name):
        """Save the etlt file.

        Args:
            model (keras.models.Model): Keras model to be saved.
            output_file_name (str): Path to the output etlt file.

        Returns:
            tmp_file_name (str): Path to the temporary uff file.
        """
        os_handle, tmp_file_name = tempfile.mkstemp()
        os.close(os_handle)
        logger.debug("Saving etlt model file at: {}.".format(output_file_name))
        input_tensor_name = ""
        # @vpraveen: commented out the preprocessor kwarg from keras_to_uff.
        # todo: @vpraveen and @zhimeng, if required modify modulus code to add
        # this.
        if self.backend == "uff":
            input_tensor_name, _, _ = keras_to_uff(
                model,
                tmp_file_name,
                output_node_names=self.output_node_names,
                custom_objects=CUSTOM_OBJS)
        else:
            raise NotImplementedError("Incompatible backend.")
        self.save_etlt(tmp_file_name, output_file_name, input_tensor_name)
        return tmp_file_name

    def save_etlt(self, tmp_file_name, output_file_name, input_tensor_name):
        """Save uff to encoded etlt file.

        Args:
            tmp_file_name (str): Path to the saved tmp uff file.
            output_file_name (str): Path to the output etlt file.
            input_tensor_name (list): List of input nodes
                Note: Currently we support only single input node
                graphs.
        """
        # Encode temporary uff to output file
        with open(tmp_file_name, "rb") as open_temp_file, open(output_file_name,
                                                               "wb") as open_encoded_file:
            # TODO: @vpraveen: Remove this hack to support multiple input nodes.
            # This will require an update to tlt_converter and DS. Postponing this for now.
            if isinstance(input_tensor_name, list):
                input_tensor_name = input_tensor_name[0]
            open_encoded_file.write(struct.pack("<i", len(input_tensor_name)))
            open_encoded_file.write(input_tensor_name.encode())
            encoding.encode(open_temp_file,
                            open_encoded_file,
                            self.key)

    def generate_ds_config(self, input_dims, num_classes=None):
        """Generate Deepstream config element for the exported model."""
        if input_dims[0] == 1:
            color_format = "l"
        else:
            color_format = "bgr" if self.preprocessing_arguments["flip_channel"] else "rgb"
        kwargs = {
            "data_format": self.data_format,
            "backend": self.backend,
            # Setting this to 0 by default because there are more
            # detection networks.
            "network_type": 0
        }
        if num_classes:
            kwargs["num_classes"] = num_classes
        if self.backend == "uff":
            kwargs.update({
                "input_names": self.input_node_names,
                "output_names": self.output_node_names
            })

        ds_config = BaseDSConfig(
            self.preprocessing_arguments["scale"],
            self.preprocessing_arguments["means"],
            input_dims,
            color_format,
            self.key,
            **kwargs
        )
        return ds_config

    def get_class_labels(self):
        """Get list of class labels to serialize to a labels.txt file."""
        return []

    def export(self, output_file_name, backend,
               calibration_cache="", data_file_name="",
               n_batches=1, batch_size=1, verbose=True,
               calibration_images_dir="", save_engine=False,
               engine_file_name="", max_workspace_size=1 << 30,
               max_batch_size=1, min_batch_size=1, opt_batch_size=1,
               force_ptq=False, static_batch_size=-1, gen_ds_config=True):
        """Simple function to export a model.

        This function sets the first converts a keras graph to uff and then saves it to an etlt
        file. After which, it verifies the parsability of the etlt file by creating a TensorRT
        engine of desired backend datatype.

        Args:
            output_file_name (str): Path to the output etlt file.
            backend (str): Backend parser to be used. ("uff", "onnx).
            calibration_cache (str): Path to the output calibration cache file.
            data_file_name (str): Path to the data tensorfile for int8 calibration.
            n_batches (int): Number of batches to calibrate model for int8 calibration.
            batch_size (int): Number of images per batch.
            verbose (bool): Flag to set verbose logging.
            calibration_images_dir (str): Path to a directory of images for custom data
                to calibrate the model over.
            save_engine (bool): Flag to save the engine after training.
            engine_file_name (str): Path to the engine file name.
            force_ptq (bool): Flag to force post training quantization using TensorRT
                for a QAT trained model. This is required iff the inference platform is
                a Jetson with a DLA.
            static_batch_size(int): Set a static batch size for exported etlt model.

        Returns:
            No explicit returns.
        """
        # set dynamic_batch flag
        dynamic_batch = bool(static_batch_size <= 0)
        # save static_batch_size for use in load_model() method
        self.static_batch_size = static_batch_size
        # Set keras session.
        self.set_backend(backend)
        self.set_input_output_node_names()
        self.status_logger.write(
            data=None, status_string=f"Using input nodes: {self.input_node_names}"
        )
        self.status_logger.write(
            data=None, status_string=f"Using output nodes: {self.output_node_names}"
        )
        logger.info("Using input nodes: {}".format(self.input_node_names))
        logger.info("Using output nodes: {}".format(self.output_node_names))

        # tensor_scale_dict is created in the load_model() method
        model = self.load_model()
        # Update parameter count to the model metadata for TAO studio.
        model_metadata = {
            "param_count": get_num_params(model)
        }

        tmp_file_name = self.save_etlt_file(model, output_file_name)
        # Add the size of the exported .etlt file for TAO Studio.
        model_metadata["size"] = get_model_file_size(output_file_name)
        # Get int8 calibrator
        calibrator = None
        input_dims = self.get_input_dims(data_file_name=data_file_name, model=model)
        max_batch_size = max(batch_size, max_batch_size)
        logger.debug("Input dims: {}".format(input_dims))
        # Clear the backend keras session.
        keras.backend.clear_session()
        if self.data_type == "int8":
            # Discard extracted tensor scales if force_ptq is set.
            if self.tensor_scale_dict is None or force_ptq:
                # no tensor scale, take traditional INT8 calibration approach
                # use calibrator to generate calibration cache
                calibrator = self.get_calibrator(calibration_cache=calibration_cache,
                                                 data_file_name=data_file_name,
                                                 n_batches=n_batches,
                                                 batch_size=batch_size,
                                                 input_dims=input_dims,
                                                 calibration_images_dir=calibration_images_dir,
                                                 image_mean=self.image_mean)
                logger.info("Calibration takes time especially if number of batches is large.")
                self.status_logger.write(
                    data=None,
                    status_string="Calibration takes time especially if number of batches is large."
                )
            else:
                # QAT model, take tensor scale approach
                # dump tensor scale to calibration cache directly
                self.status_logger.write(
                    data=None,
                    status_string="Extracting scales generated during QAT."
                )
                self._calibration_cache_from_dict(
                    self.tensor_scale_dict,
                    calibration_cache,
                )

        if gen_ds_config:
            self.set_data_preprocessing_parameters(input_dims, image_mean=self.image_mean)
            labels = self.get_class_labels()
            num_classes = None
            output_root = os.path.dirname(output_file_name)
            if labels:
                num_classes = len(labels)
                with open(os.path.join(output_root, "labels.txt"), "w") as lfile:
                    for label in labels:
                        lfile.write("{}\n".format(label))
                assert lfile.closed, (
                    "Label file wasn't closed after saving."
                )
            # Generate DS Config file.
            ds_config = self.generate_ds_config(
                input_dims,
                num_classes=num_classes
            )
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            ds_file = os.path.join(output_root, "nvinfer_config.txt")
            with open(ds_file, "w") as dsf:
                dsf.write(str(ds_config))
            assert dsf.closed, (
                "Deepstream config file wasn't closed."
            )

        # Verify with engine generation / run calibration.
        if self.backend == "uff":
            # Assuming single input node graph for uff engine creation.
            in_tensor_name = self.input_node_names[0]
            if not isinstance(input_dims, dict):
                input_dims = {in_tensor_name: input_dims}
            engine_builder = UFFEngineBuilder(
                tmp_file_name,
                in_tensor_name,
                input_dims,
                self.output_node_names,
                max_batch_size=max_batch_size,
                max_workspace_size=max_workspace_size,
                dtype=self.data_type,
                strict_type=self.strict_type,
                verbose=verbose,
                calibrator=calibrator,
                tensor_scale_dict=None if force_ptq else self.tensor_scale_dict)
            trt_engine = engine_builder.get_engine()
            if save_engine:
                with open(engine_file_name, "wb") as outf:
                    outf.write(trt_engine.serialize())
            if trt_engine:
                del trt_engine
        elif self.backend == "onnx":
            engine_builder = ONNXEngineBuilder(
                tmp_file_name,
                max_batch_size=max_batch_size,
                min_batch_size=min_batch_size,
                max_workspace_size=max_workspace_size,
                opt_batch_size=opt_batch_size,
                dtype=self.data_type,
                strict_type=self.strict_type,
                verbose=verbose,
                calibrator=calibrator,
                tensor_scale_dict=None if force_ptq else self.tensor_scale_dict,
                dynamic_batch=dynamic_batch)
            trt_engine = engine_builder.get_engine()
            if save_engine:
                with open(engine_file_name, "wb") as outf:
                    outf.write(trt_engine.serialize())
            if trt_engine:
                del trt_engine
        else:
            raise NotImplementedError("Invalid backend.")
        # Remove temporary uff file.
        os.remove(tmp_file_name)
        self.status_logger.write(
            data=model_metadata,
            status_string="Export complete."
        )
