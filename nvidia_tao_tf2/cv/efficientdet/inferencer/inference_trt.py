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
"""Standalone TensorRT inference."""
import os
import numpy as np

import pycuda.autoinit # noqa pylint: disable=unused-import
import pycuda.driver as cuda
import tensorrt as trt

from nvidia_tao_tf2.cv.efficientdet.exporter.image_batcher import ImageBatcher
from nvidia_tao_tf2.cv.efficientdet.visualize import vis_utils


class TensorRTInfer:
    """Implements inference for the EfficientDet TensorRT engine."""

    def __init__(self, engine_path,
                 label_id_mapping=None,
                 min_score_thresh=0.3):
        """Init.

        :param engine_path: The path to the serialized engine to load from disk.
        """
        self.label_id_mapping = label_id_mapping or {}
        self.min_score_thresh = min_score_thresh or 0.3
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        # Setup I/O bindings
        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0

    def input_spec(self):
        """Get the specs for the input tensor of the network. Useful to prepare memory allocations.

        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[0]['shape'], self.inputs[0]['dtype']

    def output_spec(self):
        """Get the specs for the output tensors of the network. Useful to prepare memory allocations.

        :return: A list with two items per element,
        the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for o in self.outputs:
            specs.append((o['shape'], o['dtype']))
        return specs

    def infer(self, batch, scales=None, nms_threshold=None):
        """Execute inference on a batch of images.

        The images should already be batched and preprocessed, as prepared by
        the ImageBatcher class. Memory copying to and from the GPU device will be performed here.
        :param batch: A numpy array holding the image batch.
        :param scales: The image resize scales for each image in this batch.
            Default: No scale postprocessing applied.
        :return: A nested list for each image in the batch and each detection in the list.
        """
        # Prepare the output data
        outputs = []
        for shape, dtype in self.output_spec():
            outputs.append(np.zeros(shape, dtype))

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]['allocation'], np.ascontiguousarray(batch))
        self.context.execute_v2(self.allocations)
        for o in range(len(outputs)):
            cuda.memcpy_dtoh(outputs[o], self.outputs[o]['allocation'])

        # Process the results
        nums = outputs[0]
        boxes = outputs[1]
        scores = outputs[2]
        classes = outputs[3]
        detections = []
        normalized = (np.max(boxes) < 2.0)
        for i in range(self.batch_size):
            detections.append([])
            for n in range(int(nums[i])):
                scale = self.inputs[0]['shape'][2] if normalized else 1.0
                if scales and i < len(scales):
                    scale /= scales[i]
                if nms_threshold and scores[i][n] < nms_threshold:
                    continue
                detections[i].append({
                    'ymin': boxes[i][n][0] * scale,
                    'xmin': boxes[i][n][1] * scale,
                    'ymax': boxes[i][n][2] * scale,
                    'xmax': boxes[i][n][3] * scale,
                    'score': scores[i][n],
                    'class': int(classes[i][n]),
                })
        return detections

    def __del__(self):
        """Simple function to destroy tensorrt handlers."""
        if self.context:
            del self.context

        if self.engine:
            del self.engine

    def visualize_detections(self, image_dir, output_dir, dump_label=False):
        """Visualize detection."""
        # TODO(@yuw): to use vis_utils function.
        labels = [i[1] for i in sorted(self.label_id_mapping.items(), key=lambda x: x[0])]
        batcher = ImageBatcher(image_dir, *self.input_spec())
        for batch, images, scales in batcher.get_batch():
            print(f"Processing Image {batcher.image_index} / {batcher.num_images}", end="\r")
            detections = self.infer(batch, scales, self.min_score_thresh)
            for i in range(len(images)):
                basename = os.path.splitext(os.path.basename(images[i]))[0]
                # Image Visualizations
                output_path = os.path.join(output_dir, f"{basename}.png")
                vis_utils.visualize_detections(images[i], output_path, detections[i], labels)
                if dump_label:
                    out_label_path = os.path.join(output_dir, "labels")
                    os.makedirs(out_label_path, exist_ok=True)
                    assert self.label_id_mapping, "Label mapping must be valid to generate KIITI labels."
                    # Generate KITTI labels
                    kitti_txt = ""
                    for d in detections[i]:
                        kitti_txt += self.label_id_mapping[int(d['class']) + 1] + ' 0 0 0 ' + ' '.join(
                            [str(d['xmin']), str(d['ymin']), str(d['xmax']), str(d['ymax'])]) + \
                            ' 0 0 0 0 0 0 0 ' + str(d['score']) + '\n'
                    with open(os.path.join(out_label_path, f"{basename}.txt"), "w", encoding='utf-8') as f:
                        f.write(kitti_txt)
