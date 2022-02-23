# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Copyright 2020 Google Research. All Rights Reserved.
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
# ==============================================================================
"""Inference related utilities."""

from __future__ import absolute_import
from __future__ import division
# gtype import
from __future__ import print_function

import copy
import functools
import os
import time
from typing import Any, Dict, List, Text, Tuple, Union

from absl import logging
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.client import timeline  # pylint: disable=g-direct-tensorflow-import
import yaml

from cv.efficientdet.dataloader import dataloader
from cv.efficientdet.model import anchors
from cv.efficientdet.model import efficientdet
from cv.efficientdet.utils import hparams_config
from cv.efficientdet.utils import keras_utils, model_utils
from cv.efficientdet.visualize import vis_utils

coco_id_mapping = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush',
}  # pyformat: disable


def image_preprocess(image, image_size: Union[int, Tuple[int, int]]):
    """Preprocess image for inference.

    Args:
        image: input image, can be a tensor or a numpy arary.
        image_size: single integer of image size for square image or tuple of two
            integers, in the format of (image_height, image_width).

    Returns:
        (image, scale): a tuple of processed image and its scale.
    """
    input_processor = dataloader.DetectionInputProcessor(image, image_size)
    input_processor.normalize_image()
    input_processor.set_scale_factors_to_output_size()
    image = input_processor.resize_and_crop_image()
    image_scale = input_processor.image_scale_to_original
    return image, image_scale


def batch_image_files_decode(image_files):
    """Decode batch of images."""
    def decode(image_file):
        image = tf.io.decode_image(image_file)
        image.set_shape([None, None, None])
        return image

    raw_images = tf.map_fn(decode, image_files, dtype=tf.uint8)
    return tf.stack(raw_images)


def batch_image_preprocess(raw_images,
                           image_size: Union[int, Tuple[int, int]],
                           batch_size: int = None):
    """Preprocess batched images for inference.

    Args:
        raw_images: a list of images, each image can be a tensor or a numpy arary.
        image_size: single integer of image size for square image or tuple of two
            integers, in the format of (image_height, image_width).
        batch_size: if None, use map_fn to deal with dynamic batch size.

    Returns:
        (image, scale): a tuple of processed images and scales.
    """
    if not batch_size:
        # map_fn is a little bit slower due to some extra overhead.
        map_fn = functools.partial(image_preprocess, image_size=image_size)
        images, scales = tf.map_fn(
            map_fn, raw_images, dtype=(tf.float32, tf.float32), back_prop=False)
        return (images, scales)

    # If batch size is known, use a simple loop.
    scales, images = [], []
    for i in range(batch_size):
        image, scale = image_preprocess(raw_images[i], image_size)
        scales.append(scale)
        images.append(image)
    images = tf.stack(images)
    scales = tf.stack(scales)
    return (images, scales)


def build_inputs(image_path_pattern: Text, image_size: Union[int, Tuple[int, int]]):
    """Read and preprocess input images.

    Args:
        image_path_pattern: a path to indicate a single or multiple files.
        image_size: single integer of image size for square image or tuple of two
            integers, in the format of (image_height, image_width).

    Returns:
        (raw_images, images, scales): raw images, processed images, and scales.

    Raises:
        ValueError if image_path_pattern doesn't match any file.
    """
    raw_images, fnames = [], []
    for fname in tf.io.gfile.glob(image_path_pattern):
        image = Image.open(fname).convert('RGB')
        raw_images.append(image)
        fnames.append(fname)
    if not raw_images:
        raise ValueError(
                'Cannot find any images for pattern {}'.format(image_path_pattern))
    return raw_images, fnames


def det_post_process_combined(params, cls_outputs, box_outputs, scales,
                              min_score_thresh, max_boxes_to_draw):
    """A combined version of det_post_process with dynamic batch size support."""
    batch_size = tf.shape(list(cls_outputs)[0])[0]

    cls_outputs_all = []
    box_outputs_all = []
    # Concatenates class and box of all levels into one tensor.
    for level in range(0, params['max_level'] - params['min_level'] + 1):
        if params['data_format'] == 'channels_first':
            cls_outputs[level] = tf.transpose(cls_outputs[level], [0, 2, 3, 1])
            box_outputs[level] = tf.transpose(box_outputs[level], [0, 2, 3, 1])

        cls_outputs_all.append(
            tf.reshape(cls_outputs[level], [batch_size, -1, params['num_classes']]))
        box_outputs_all.append(
            tf.reshape(box_outputs[level], [batch_size, -1, 4]))
    cls_outputs_all = tf.concat(cls_outputs_all, 1)
    box_outputs_all = tf.concat(box_outputs_all, 1)

    # Create anchor_label for picking top-k predictions.
    eval_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                   params['num_scales'], params['aspect_ratios'],
                                   params['anchor_scale'], params['image_size'])
    anchor_boxes = eval_anchors.boxes
    scores = tf.math.sigmoid(cls_outputs_all)
    # apply bounding box regression to anchors
    boxes = anchors.decode_box_outputs_tf(box_outputs_all, anchor_boxes)
    boxes = tf.expand_dims(boxes, axis=2)
    scales = tf.expand_dims(scales, axis=-1)
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = (
        tf.image.combined_non_max_suppression(
            boxes,
            scores,
            max_boxes_to_draw,
            max_boxes_to_draw,
            score_threshold=min_score_thresh,
            clip_boxes=False))
    del valid_detections  # to be used in future.

    image_ids = tf.cast(
        tf.tile(
            tf.expand_dims(tf.range(batch_size), axis=1), [1, max_boxes_to_draw]),
        dtype=tf.float32)
    image_size = model_utils.parse_image_size(params['image_size'])
    ymin = tf.clip_by_value(nmsed_boxes[..., 0], 0, image_size[0]) * scales
    xmin = tf.clip_by_value(nmsed_boxes[..., 1], 0, image_size[1]) * scales
    ymax = tf.clip_by_value(nmsed_boxes[..., 2], 0, image_size[0]) * scales
    xmax = tf.clip_by_value(nmsed_boxes[..., 3], 0, image_size[1]) * scales

    classes = tf.cast(nmsed_classes + 1, tf.float32)
    detection_list = [image_ids, ymin, xmin, ymax, xmax, nmsed_scores, classes]
    detections = tf.stack(detection_list, axis=2, name='detections')
    return detections


def visualize_image(image,
                    boxes,
                    classes,
                    scores,
                    id_mapping,
                    min_score_thresh=anchors.MIN_SCORE_THRESH,
                    max_boxes_to_draw=anchors.MAX_DETECTIONS_PER_IMAGE,
                    line_thickness=2,
                    **kwargs):
    """Visualizes a given image.

    Args:
        image: a image with shape [H, W, C].
        boxes: a box prediction with shape [N, 4] ordered [ymin, xmin, ymax, xmax].
        classes: a class prediction with shape [N].
        scores: A list of float value with shape [N].
        id_mapping: a dictionary from class id to name.
        min_score_thresh: minimal score for showing. If claass probability is below
            this threshold, then the object will not show up.
        max_boxes_to_draw: maximum bounding box to draw.
        line_thickness: how thick is the bounding box line.
        **kwargs: extra parameters.

    Returns:
        output_image: an output image with annotated boxes and classes.
    """
    category_index = {k: {'id': k, 'name': id_mapping[k]} for k in id_mapping}
    img = np.array(image)
    vis_utils.visualize_boxes_and_labels_on_image_array(
        img,
        boxes,
        classes,
        scores,
        category_index,
        min_score_thresh=min_score_thresh,
        max_boxes_to_draw=max_boxes_to_draw,
        line_thickness=line_thickness,
        **kwargs)
    return img


def parse_label_id_mapping(label_id_mapping):
    """Parse label id mapping from a string or a yaml file.

    The label_id_mapping is a dict that maps class id to its name, such as:

        {
            1: "person",
            2: "dog"
        }

    Args:
        label_id_mapping:

    Returns:
        A dictionary with key as integer id and value as a string of name.
    """
    if label_id_mapping is None:
        return coco_id_mapping

    if isinstance(label_id_mapping, dict):
        label_id_dict = label_id_mapping
    elif isinstance(label_id_mapping, str):
        with tf.io.gfile.GFile(label_id_mapping) as f:
            label_id_dict = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise TypeError('label_id_mapping must be a dict or a yaml filename, '
                        'containing a mapping from class ids to class names.')

    return label_id_dict


def visualize_image_prediction(image,
                               prediction,
                               disable_pyfun=True,
                               label_id_mapping=None,
                               **kwargs):
    """Viusalize detections on a given image.

    Args:
        image: Image content in shape of [height, width, 3].
        prediction: a list of vector, with each vector has the format of [image_id,
            ymin, xmin, ymax, xmax, score, class].
        disable_pyfun: disable pyfunc for faster post processing.
        label_id_mapping: a map from label id to name.
        **kwargs: extra parameters for vistualization, such as min_score_thresh,
            max_boxes_to_draw, and line_thickness.

    Returns:
        a list of annotated images.
    """
    boxes = prediction[:, 1:5]
    classes = prediction[:, 6].astype(int)
    scores = prediction[:, 5]

    if not disable_pyfun:
        # convert [x, y, width, height] to [y, x, height, width]
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

    label_id_mapping = label_id_mapping or {}  # coco_id_mapping

    return visualize_image(image, boxes, classes, scores, label_id_mapping,
                           **kwargs)


class InferenceModel(tf.Module): 
    def __init__(self, model, input_shape, params,
                 batch_size=1, label_id_mapping=None,
                 min_score_thresh=0.3, max_boxes_to_draw=100):
        super().__init__()
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.params = params
        self.disable_pyfun = True
        self.label_id_mapping = label_id_mapping or {}
        self.min_score_thresh = min_score_thresh or 0.3
        self.max_boxes_to_draw = max_boxes_to_draw or 100

    def infer(self, imgs):
        images, scales = batch_image_preprocess(imgs, self.input_shape, self.batch_size)
        cls_outputs, box_outputs = self.model(images, training=False)

        detections = det_post_process_combined(
            self.params,
            cls_outputs, box_outputs,
            scales,
            min_score_thresh=self.min_score_thresh,
            max_boxes_to_draw=self.max_boxes_to_draw)
        return detections

    def visualize_detections(self, image_paths, output_dir, dump_label=False, **kwargs):
        raw_images, fnames = build_inputs(image_paths, self.input_shape)
        self.batch_size = min(len(raw_images), self.batch_size)
        if self.batch_size < 1:
            return
        # run inference and render annotations
        predictions = np.array(self.infer(raw_images))
        for i, prediction in enumerate(predictions):
            img = visualize_image_prediction(
                raw_images[i],
                prediction,
                disable_pyfun=self.disable_pyfun,
                label_id_mapping=self.label_id_mapping,
                **kwargs)
            output_image_path = os.path.join(output_dir, os.path.basename(fnames[i]))
            Image.fromarray(img).save(output_image_path)
            logging.info('writing output image to %s', output_image_path)
            if dump_label:
                out_label_path = os.path.join(output_dir, 'labels')
                assert self.label_id_mapping, \
                    "Label mapping must be valid to generate KIITI labels."
                os.makedirs(out_label_path, exist_ok=True)
                # Generate KITTI labels
                kitti_txt = ""
                for d in prediction:
                    if d[5] >= kwargs.get('min_score_thresh', 0):
                        kitti_txt += self.label_id_mapping[int(d[6])] + ' 0 0 0 ' + ' '.join(
                            [str(i) for i in [d[2], d[1], d[4], d[3]]]) + ' 0 0 0 0 0 0 0 ' + \
                                str(d[5]) + '\n'
                basename = os.path.splitext(os.path.basename(fnames[i]))[0]
                with open(os.path.join(out_label_path, "{}.txt".format(basename)), "w") as f:
                    f.write(kitti_txt)
