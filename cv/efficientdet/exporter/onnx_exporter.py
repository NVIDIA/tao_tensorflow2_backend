# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
"""EfficientDet ONNX exporter."""

import copy
import logging
import os
import tempfile

import numpy as np
import onnx
from onnx import numpy_helper
from onnx import shape_inference
import onnx_graphsurgeon as gs

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_util
from tf2onnx import optimizer, tf_loader, tf_utils, tfonnx
from tf2onnx.utils import make_sure

from cv.efficientdet.exporter import onnx_utils # noqa pylint: disable=W0611

logging.basicConfig(level=logging.INFO)
logging.getLogger("EfficientDetGraphSurgeon").setLevel(logging.INFO)
log = logging.getLogger("EfficientDetGraphSurgeon")


def get_tf_tensor_data(tensor):
    """Get data from tensor."""
    make_sure(isinstance(tensor, tensor_pb2.TensorProto), "Require TensorProto")
    np_data = tensor_util.MakeNdarray(tensor)
    make_sure(isinstance(np_data, np.ndarray), "%r isn't ndarray", np_data)
    return np_data


def tf_to_onnx_tensor(tensor, name=""):
    """Convert tensorflow tensor to onnx tensor."""
    np_data = get_tf_tensor_data(tensor)
    if np_data.dtype == np.object:
        # assume np_data is string, numpy_helper.from_array accepts ndarray,
        # in which each item is of str while the whole dtype is of object.
        try:
            # Faster but fails on Unicode
            # np_data = np_data.astype(np.str).astype(np.object)
            if len(np_data.shape) > 0:
                np_data = np_data.astype(np.str).astype(np.object)
            else:
                np_data = np.array(str(np_data)).astype(np.object)
        except UnicodeDecodeError:
            decode = np.vectorize(lambda x: x.decode('UTF-8'))
            np_data = decode(np_data).astype(np.object)
        except Exception as e:  # noqa pylint: disable=W0611
            raise RuntimeError(f"Not support type: {type(np_data.flat[0])}") from e
    return numpy_helper.from_array(np_data, name=name)


tf_utils.tf_to_onnx_tensor = tf_to_onnx_tensor


@gs.Graph.register()
def replace_with_reducemean(self, inputs, outputs):
    """Replace subgraph with ReduceMean."""
    # Disconnect output nodes of all input tensors
    new_outputs = outputs.copy()
    # Disconnet input nodes of all output tensors
    for out in outputs:
        out.inputs.clear()
    # Insert the new node.
    return self.layer(op="ReduceMean", inputs=inputs, outputs=new_outputs, attrs={'axes': [2, 3], 'keepdims': 1})


class EfficientDetGraphSurgeon:
    """EfficientDet GraphSurgeon Class."""

    def __init__(self, saved_model_path, legacy_plugins=False, dynamic_batch=False, is_qat=False):
        """Constructor of the EfficientDet Graph Surgeon object.

        :param saved_model_path: The path pointing to the TensorFlow saved model to load.
        :param legacy_plugins: If using TensorRT version < 8.0.1,
        set this to True to use older (but slower) plugins.
        """
        saved_model_path = os.path.realpath(saved_model_path)
        assert os.path.exists(saved_model_path)

        # Let TensorRT optimize QDQ nodes instead of TF
        from tf2onnx.optimizer import _optimizers  # noqa pylint: disable=C0415
        updated_optimizers = copy.deepcopy(_optimizers)
        del updated_optimizers["q_dq_optimizer"]
        del updated_optimizers["const_dequantize_optimizer"]

        # Use tf2onnx to convert saved model to an initial ONNX graph.
        graph_def, inputs, outputs = tf_loader.from_saved_model(
            saved_model_path, None, None, "serve", ["serving_default"])
        print(f"Loaded saved model from {saved_model_path}")
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name="")
        with tf_loader.tf_session(graph=tf_graph):
            onnx_graph = tfonnx.process_tf_graph(
                tf_graph, input_names=inputs, output_names=outputs, opset=13)
        onnx_model = optimizer.optimize_graph(onnx_graph, optimizers=updated_optimizers).make_model(
            f"Converted from {saved_model_path}")
        self.graph = gs.import_onnx(onnx_model)
        assert self.graph
        log.info("TF2ONNX graph created successfully")
        self.is_qat = is_qat

        # Fold constants via ONNX-GS that TF2ONNX may have missed
        if not self.is_qat:
            self.graph.fold_constants()
        self.batch_size = None
        self.dynamic_batch = dynamic_batch
        self.legacy_plugins = legacy_plugins
        os_handle, self.tmp_onnx_path = tempfile.mkstemp(suffix='.onnx', dir=saved_model_path)
        os.close(os_handle)

    def infer(self):
        """Sanitize the graph by cleaning any unconnected nodes.

        do a topological resort and fold constant inputs values.
        When possible, run shape inference on the ONNX graph to determine tensor shapes.
        """
        for _ in range(3):
            count_before = len(self.graph.nodes)

            self.graph.cleanup().toposort()
            try:
                for node in self.graph.nodes:
                    for o in node.outputs:
                        o.shape = None
                model = gs.export_onnx(self.graph)
                model = shape_inference.infer_shapes(model)
                self.graph = gs.import_onnx(model)
            except Exception as e:
                raise RuntimeError("Shape inference could not be performed at this time") from e
            if not self.is_qat:
                try:
                    self.graph.fold_constants(fold_shapes=True)
                except TypeError as e:
                    raise TypeError("This version of ONNX GraphSurgeon does not support folding shapes, "
                                    "please upgrade your onnx_graphsurgeon module.") from e

            count_after = len(self.graph.nodes)
            if count_before == count_after:
                # No new folding occurred in this iteration, so we can stop for now.
                break

    def save(self, output_path=None):
        """Save the ONNX model to the given location.

        :param output_path: Path pointing to the location where to
        write out the updated ONNX model.
        """
        self.graph.cleanup().toposort()
        model = gs.export_onnx(self.graph)
        output_path = output_path or self.tmp_onnx_path
        onnx.save(model, output_path)
        return output_path

    def update_preprocessor(self, input_format, input_size, preprocessor="imagenet"):
        """Remove all the pre-processing nodes in the ONNX graph and leave only the image normalization essentials.

        :param input_format: The input data format, either "NCHW" or "NHWC".
        :param input_size: The input size as a comma-separated string in H,W format, e.g. "512,512".
        :param preprocessor: The preprocessor to use, either "imagenet" for imagenet mean and stdev normalization,
        or "scale_range" for uniform [-1,+1] range normalization.
        """
        # Update the input and output tensors shape
        # input_size = input_size.split(",")
        assert len(input_size) == 4
        assert input_format in ["NCHW", "NHWC"]

        if self.dynamic_batch:
            # Enable dynamic batchsize
            if input_format == "NCHW":
                self.graph.inputs[0].shape = ['N', 3, input_size[2], input_size[3]]
            if input_format == "NHWC":
                self.graph.inputs[0].shape = ['N', input_size[1], input_size[2], 3]
        else:
            # Disable dynamic batchsize
            self.graph.inputs[0].shape = input_size
        self.graph.inputs[0].dtype = np.float32
        self.graph.inputs[0].name = "input"
        print(f"ONNX graph input shape: {self.graph.inputs[0].shape} [{input_format} format]")
        self.infer()

        # Find the initial nodes of the graph, whatever the input is first connected to, and disconnect them
        for node in [node for node in self.graph.nodes if self.graph.inputs[0] in node.inputs]:
            node.inputs.clear()

        # Convert to NCHW format if needed
        input_tensor = self.graph.inputs[0]
        if input_format == "NHWC":
            input_tensor = self.graph.transpose("preprocessor/transpose", input_tensor, [0, 3, 1, 2])

        assert preprocessor in ["imagenet", "scale_range"]
        preprocessed_tensor = None
        if preprocessor == "imagenet":
            # RGB Normalizers. The per-channel values are given with shape [1, 3, 1, 1] for proper NCHW shape broadcasting
            scale_val = 1 / np.asarray([255], dtype=np.float32)
            mean_val = -1 * np.expand_dims(np.asarray([0.485, 0.456, 0.406], dtype=np.float32), axis=(0, 2, 3))
            stddev_val = 1 / np.expand_dims(np.asarray([0.224, 0.224, 0.224], dtype=np.float32), axis=(0, 2, 3))
            # y = (x * scale + mean) * stddev   -->   y = x * scale * stddev + mean * stddev
            scale_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, scale_val * stddev_val)
            mean_out = self.graph.elt_const("Add", "preprocessor/mean", scale_out, mean_val * stddev_val)
            preprocessed_tensor = mean_out[0]
        if preprocessor == "scale_range":
            # RGB Normalizers. The per-channel values are given with shape [1, 3, 1, 1] for proper NCHW shape broadcasting
            scale_val = 2 / np.asarray([255], dtype=np.float32)
            offset_val = np.expand_dims(np.asarray([-1, -1, -1], dtype=np.float32), axis=(0, 2, 3))
            # y = (x * scale + mean) * stddev   -->   y = x * scale * stddev + mean * stddev
            scale_out = self.graph.elt_const("Mul", "preprocessor/scale", input_tensor, scale_val)
            range_out = self.graph.elt_const("Add", "preprocessor/range", scale_out, offset_val)
            preprocessed_tensor = range_out[0]

        # Find the first stem conv node of the graph, and connect the normalizer directly to it
        stem_name = "stem_conv"

        # Remove transpose in QAT graph: stem <- transpose <- DQ <- Q
        if self.is_qat:
            print("Removing QAT transpose")
            transpose_node = [node for node in self.graph.nodes if node.op == "Transpose" and stem_name in node.o().name][0]
            dq_node = transpose_node.i()
            dq_node.outputs = transpose_node.outputs
            transpose_node.outputs.clear()

        stem = [node for node in self.graph.nodes
                if node.op == "Conv" and stem_name in node.name][0]
        print(f"Found {stem.op} node '{stem.name}' as stem entry")

        if self.is_qat:
            # # stem <- DQ <- Q
            # stem.i().i().i().outputs[0].dtype = np.uint8
            stem.i().i().inputs[0] = preprocessed_tensor
        else:
            stem.inputs[0] = preprocessed_tensor

        # Patch for QAT export (@yuw)
        if 'auto_pad' not in stem.attrs:
            stem.attrs['auto_pad'] = 'NOTSET'
            stem.attrs['pads'] = [0, 0, 1, 1]

        self.infer()

    def update_shapes(self):
        """Update shapes."""
        # Reshape nodes have the batch dimension as a fixed value of 1, they should use the batch size instead
        # Output-Head reshapes use [1, -1, C], corrected reshape value should be [-1, V, C]
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            shape_in = node.inputs[0].shape
            if shape_in is None or len(shape_in) not in [4, 5]:  # TFOD graphs have 5-dim inputs on this Reshape
                continue
            if type(node.inputs[1]) != gs.Constant:
                continue
            shape_out = node.inputs[1].values
            if len(shape_out) != 3 or shape_out[0] != 1 or shape_out[1] != -1:
                continue
            volume = shape_in[1] * shape_in[2] * shape_in[3] / shape_out[2]
            if len(shape_in) == 5:
                volume *= shape_in[4]
            shape_corrected = np.asarray([-1, volume, shape_out[2]], dtype=np.int64)
            node.inputs[1] = gs.Constant(f"{node.name}_shape", values=shape_corrected)
            print(f"Updating Output-Head Reshape node {node.name} to {node.inputs[1].values}")

        # Other Reshapes only need to change the first dim to -1, as long as there are no -1's already
        for node in [node for node in self.graph.nodes if node.op == "Reshape"]:
            if type(node.inputs[1]) != gs.Constant or node.inputs[1].values[0] != 1 or -1 in node.inputs[1].values:
                continue
            node.inputs[1].values[0] = -1
            print(f"Updating Reshape node {node.name} to {node.inputs[1].values}")

        # Resize nodes try to calculate the output shape dynamically, it's more optimal to pre-compute the shape
        # Resize on a BiFPN will always be 2x, but grab it from the graph just in case
        for node in [node for node in self.graph.nodes if node.op == "Resize"]:
            if len(node.inputs) < 4 or node.inputs[0].shape is None:
                continue
            scale_h, scale_w = None, None
            if type(node.inputs[3]) == gs.Constant:
                # The sizes input is already folded
                if len(node.inputs[3].values) != 4:
                    continue
                scale_h = node.inputs[3].values[2] / node.inputs[0].shape[2]
                scale_w = node.inputs[3].values[3] / node.inputs[0].shape[3]
            if type(node.inputs[3]) == gs.Variable:
                # The sizes input comes from Shape+Slice+Concat
                concat = node.i(3)
                if concat.op != "Concat":
                    continue
                if type(concat.inputs[1]) != gs.Constant or len(concat.inputs[1].values) != 2:
                    continue
                scale_h = concat.inputs[1].values[0] / node.inputs[0].shape[2]
                scale_w = concat.inputs[1].values[1] / node.inputs[0].shape[3]
            scales = np.asarray([1, 1, scale_h, scale_w], dtype=np.float32)
            del node.inputs[3]
            node.inputs[2] = gs.Constant(name=f"{node.name}_scales", values=scales)
            print(f"Updating Resize node {node.name} to {scales}")

        self.infer()

    def update_network(self):
        """Updates the graph.

        To replace certain nodes in the main EfficientDet network
        """
        # EXPERIMENTAL
        # for node in self.graph.nodes:
        #     if node.op == "GlobalAveragePool" and node.o().op == "Squeeze" and node.o().o().op == "Reshape":
        #         # node Mul has two output nodes: GlobalAveragePool and Mul
        #         # we only disconnect GlobalAveragePool
        #         self.graph.replace_with_reducemean(node.inputs, node.o().o().outputs)
        #         print("Pooling removed.")
        # self.graph.cleanup().toposort()
        pass

    def update_nms(self, threshold=None, detections=None):
        """Updates the graph to replace the NMS op by BatchedNMS_TRT TensorRT plugin node.

        :param threshold: Override the score threshold attribute. If set to None,
        use the value in the graph.
        :param detections: Override the max detections attribute. If set to None,
        use the value in the graph.
        """

        def find_head_concat(name_scope):
            # This will find the concatenation node at the end of either Class Net or Box Net.
            # These concatenation nodes bring together prediction data for each of 5 scales.
            # The concatenated Class Net node will have shape
            # [batch_size, num_anchors, num_classes],
            # and the concatenated Box Net node has the shape [batch_size, num_anchors, 4].
            # These concatenation nodes can be be found by searching for all Concat's
            # and checking if the node two steps above in the graph has a name that begins with
            # either "box_net/..." or "class_net/...".
            for node in [node for node in self.graph.nodes
                         if node.op == "Transpose" and name_scope in node.name]:
                concat = self.graph.find_descendant_by_op(node, "Concat")
                assert concat and len(concat.inputs) == 5
                log.info("Found {} node '{}' as the tip of {}".format(  # noqa pylint: disable=C0209
                    concat.op, concat.name, name_scope))
                return concat

        def extract_anchors_tensor(split):
            # This will find the anchors that have been hardcoded somewhere within the ONNX graph.
            # The function will return a gs.Constant that can be directly used as
            # an input to the NMS plugin.
            # The anchor tensor shape will be [1, num_anchors, 4].
            # Note that '1' is kept as first dim, regardless of batch size,
            # as it's not necessary to replicate the anchors for all images in the batch.

            # The anchors are available (one per coordinate) hardcoded as constants
            # within certain box decoder nodes.
            # Each of these four constants have shape [1, num_anchors], so some numpy operations
            # are used to expand the dims and concatenate them as needed.

            # These constants can be found by starting from the Box Net's split operation,
            # and for each coordinate, walking down in the graph until either an Add or
            # Mul node is found. The second input on this nodes will be the anchor data required.
            def get_anchor_np(output_idx, op):
                node = self.graph.find_descendant_by_op(split.o(0, output_idx), op)
                assert node
                val = np.squeeze(node.inputs[1].values)
                return np.expand_dims(val.flatten(), axis=(0, 2))

            anchors_y = get_anchor_np(0, "Add")
            anchors_x = get_anchor_np(1, "Add")
            anchors_h = get_anchor_np(2, "Mul")
            anchors_w = get_anchor_np(3, "Mul")
            anchors = np.concatenate([anchors_y, anchors_x, anchors_h, anchors_w], axis=2)
            return gs.Constant(name="nms/anchors:0", values=anchors)

        self.infer()

        head_names = ["class-predict", "box-predict"]

        # There are five nodes at the bottom of the graph that provide important connection points:

        # 1. Find the concat node at the end of the class net (multi-scale class predictor)
        class_net = find_head_concat(head_names[0])
        class_net_tensor = class_net.outputs[0]

        # 2. Find the concat node at the end of the box net (multi-scale localization predictor)
        box_net = find_head_concat(head_names[1])
        box_net_tensor = box_net.outputs[0]

        # 3. Find the split node that separates the box net coordinates
        # and feeds them into the box decoder.
        box_net_split = self.graph.find_descendant_by_op(box_net, "Split")
        assert box_net_split and len(box_net_split.outputs) == 4

        # 4. Find the concat node at the end of the box decoder.
        box_decoder = self.graph.find_descendant_by_op(box_net_split, "Concat")
        assert box_decoder and len(box_decoder.inputs) == 4
        # box_decoder_tensor = box_decoder.outputs[0]

        # 5. Find the NMS node.
        nms_node = self.graph.find_node_by_op("NonMaxSuppression")

        # Extract NMS Configuration
        num_detections = int(nms_node.inputs[2].values) if detections is None else detections
        iou_threshold = float(nms_node.inputs[3].values)
        score_threshold = float(nms_node.inputs[4].values) if threshold is None else threshold
        # num_classes = class_net.i().inputs[1].values[-1]
        # normalized = False

        # NMS Inputs and Attributes
        # NMS expects these shapes for its input tensors:
        # box_net: [batch_size, number_boxes, 4]
        # class_net: [batch_size, number_boxes, number_classes]
        # anchors: [1, number_boxes, 4] (if used)
        nms_op = None
        nms_attrs = None
        nms_inputs = None

        # EfficientNMS TensorRT Plugin
        # Fusing the decoder will always be faster, so this is
        # the default NMS method supported. In this case,
        # three inputs are given to the NMS TensorRT node:
        # - The box predictions (from the Box Net node found above)
        # - The class predictions (from the Class Net node found above)
        # - The default anchor coordinates (from the extracted anchor constants)
        # As the original tensors from EfficientDet will be used,
        # the NMS code type is set to 1 (Center+Size),
        # because this is the internal box coding format used by the network.
        anchors_tensor = extract_anchors_tensor(box_net_split)
        nms_inputs = [box_net_tensor, class_net_tensor, anchors_tensor]
        nms_op = "EfficientNMS_TRT"
        nms_attrs = {
            'plugin_version': "1",
            'background_class': -1,
            'max_output_boxes': num_detections,
            # Keep threshold to at least 0.01 for better efficiency
            'score_threshold': max(0.01, score_threshold),
            'iou_threshold': iou_threshold,
            'score_activation': True,
            'box_coding': 1,
        }
        nms_output_classes_dtype = np.int32

        # NMS Outputs
        if self.dynamic_batch:
            # Enable dynamic batch
            nms_output_num_detections = gs.Variable(
                name="num_detections", dtype=np.int32, shape=['N', 1])
            nms_output_boxes = gs.Variable(
                name="detection_boxes", dtype=np.float32, shape=['N', num_detections, 4])
            nms_output_scores = gs.Variable(
                name="detection_scores", dtype=np.float32, shape=['N', num_detections])
            nms_output_classes = gs.Variable(
                name="detection_classes", dtype=nms_output_classes_dtype, shape=['N', num_detections])
        else:
            nms_output_num_detections = gs.Variable(
                name="num_detections", dtype=np.int32, shape=[self.batch_size, 1])
            nms_output_boxes = gs.Variable(
                name="detection_boxes", dtype=np.float32, shape=[self.batch_size, num_detections, 4])
            nms_output_scores = gs.Variable(
                name="detection_scores", dtype=np.float32, shape=[self.batch_size, num_detections])
            nms_output_classes = gs.Variable(
                name="detection_classes", dtype=nms_output_classes_dtype, shape=[self.batch_size, num_detections])

        nms_outputs = [
            nms_output_num_detections,
            nms_output_boxes,
            nms_output_scores,
            nms_output_classes]

        # Create the NMS Plugin node with the selected inputs.
        # The outputs of the node will also become the final outputs of the graph.
        self.graph.plugin(
            op=nms_op,
            name="nms/non_maximum_suppression",
            inputs=nms_inputs,
            outputs=nms_outputs,
            attrs=nms_attrs)
        log.info("Created NMS plugin '{}' with attributes: {}".format(nms_op, nms_attrs))  # noqa pylint: disable=C0209

        self.graph.outputs = nms_outputs

        self.infer()
