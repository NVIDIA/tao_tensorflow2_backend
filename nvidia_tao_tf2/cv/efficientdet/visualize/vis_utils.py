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
"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.
"""
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from nvidia_tao_tf2.cv.efficientdet.visualize import shape_utils
from nvidia_tao_tf2.cv.efficientdet.visualize import standard_fields as fields

matplotlib.use('Agg')  # Set headless-friendly backend.
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def _get_multiplier_for_color_randomness():
    """Returns a multiplier to get semi-random colors from successive indices.

    This function computes a prime number, p, in the range [2, 17] that:
    - is closest to len(STANDARD_COLORS) / 10
    - does not divide len(STANDARD_COLORS)

    If no prime numbers in that range satisfy the constraints, p is returned as 1.

    Once p is established, it can be used as a multiplier to select
    non-consecutive colors from STANDARD_COLORS:
    colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
    """
    num_colors = len(STANDARD_COLORS)
    prime_candidates = [5, 7, 11, 13, 17]

    # Remove all prime candidates that divide the number of colors.
    prime_candidates = [p for p in prime_candidates if num_colors % p]
    if not prime_candidates:
        return 1

    # Return the closest prime number to num_colors / 10.
    abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
    num_candidates = len(abs_distance)
    inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
    return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
    """Saves an image (represented as a numpy array) to PNG.

    Args:
        image: a numpy array with shape [height, width, 3].
        output_path: path to which image should be written.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.gfile.Open(output_path, 'w') as fid:
        image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
    """Encodes a numpy array into a PNG string.

    Args:
        image: a numpy array with shape [height, width, 3].

    Returns:
        PNG encoded image string.
    """
    image_pil = Image.fromarray(np.uint8(image))
    output = six.BytesIO()
    image_pil.save(output, format='PNG')
    png_string = output.getvalue()
    output.close()
    return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
    """Adds a bounding box to an image (numpy array).

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Args:
        image: a numpy array with shape [height, width, 3].
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box (each to be shown on its
            own line).
        use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
            ymax, xmax as relative to the image.  Otherwise treat coordinates as
            absolute.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                               thickness, display_str_list,
                               use_normalized_coordinates)
    np.copyto(image, np.array(image_pil))


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
        image: a PIL.Image object.
        ymin: ymin of bounding box.
        xmin: xmin of bounding box.
        ymax: ymax of bounding box.
        xmax: xmax of bounding box.
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list: list of strings to display in box (each to be shown on its
            own line).
        use_normalized_coordinates: If True (default), treat coordinates ymin, xmin,
            ymax, xmax as relative to the image.  Otherwise treat coordinates as
            absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    if thickness > 0:
        draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
                  width=thickness,
                  fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width = font.getbbox(display_str)[2] - font.getbbox(display_str)[0]
        text_height = font.getbbox(display_str)[3] - font.getbbox(display_str)[1]
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)],
                       fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

    Args:
        image: a numpy array object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
            coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings. a list of strings for each
            bounding box. The reason to pass a list of strings for a bounding box is
            that it might contain multiple labels.

    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.

    Args:
        image: a PIL.Image object.
        boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax). The
            coordinates are in normalized format between [0, 1].
        color: color to draw bounding box. Default is red.
        thickness: line thickness. Default value is 4.
        display_str_list_list: list of list of strings. a list of strings for each
            bounding box. The reason to pass a list of strings for a bounding box is
            that it might contain multiple labels.

    Raises:
        ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness, display_str_list)


def create_visualization_fn(category_index,
                            include_masks=False,
                            include_keypoints=False,
                            include_track_ids=False,
                            **kwargs):
    """Constructs a visualization function that can be wrapped in a py_func.

    py_funcs only accept positional arguments. This function returns a suitable
    function with the correct positional argument mapping. The positional
    arguments in order are:
    0: image
    1: boxes
    2: classes
    3: scores
    [4-6]: masks (optional)
    [4-6]: keypoints (optional)
    [4-6]: track_ids (optional)

    -- Example 1 --
    vis_only_masks_fn = create_visualization_fn(category_index,
        include_masks=True, include_keypoints=False, include_track_ids=False,
        **kwargs)
    image = tf.py_func(vis_only_masks_fn,
                                         inp=[image, boxes, classes, scores, masks],
                                         Tout=tf.uint8)

    -- Example 2 --
    vis_masks_and_track_ids_fn = create_visualization_fn(category_index,
        include_masks=True, include_keypoints=False, include_track_ids=True,
        **kwargs)
    image = tf.py_func(vis_masks_and_track_ids_fn,
                                         inp=[image, boxes, classes, scores, masks, track_ids],
                                         Tout=tf.uint8)

    Args:
        category_index: a dict that maps integer ids to category dicts. e.g.
            {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
        include_masks: Whether masks should be expected as a positional argument in
            the returned function.
        include_keypoints: Whether keypoints should be expected as a positional
            argument in the returned function.
        include_track_ids: Whether track ids should be expected as a positional
            argument in the returned function.
        **kwargs: Additional kwargs that will be passed to
            visualize_boxes_and_labels_on_image_array.

    Returns:
        Returns a function that only takes tensors as positional arguments.
    """

    def visualization_py_func_fn(*args):
        """Visualization function that can be wrapped in a tf.py_func.

        Args:
            *args: First 4 positional arguments must be: image - uint8 numpy array
                with shape (img_height, img_width, 3). boxes - a numpy array of shape
                [N, 4]. classes - a numpy array of shape [N]. scores - a numpy array of
                shape [N] or None. -- Optional positional arguments -- instance_masks -
                a numpy array of shape [N, image_height, image_width]. keypoints - a
                numpy array of shape [N, num_keypoints, 2]. track_ids - a numpy array of
                shape [N] with unique track ids.

        Returns:
            uint8 numpy array with shape (img_height, img_width, 3) with overlaid
            boxes.
        """
        image = args[0]
        boxes = args[1]
        classes = args[2]
        scores = args[3]
        masks = keypoints = track_ids = None
        pos_arg_ptr = 4  # Positional argument for first optional tensor (masks).
        if include_masks:
            masks = args[pos_arg_ptr]
            pos_arg_ptr += 1
        if include_keypoints:
            keypoints = args[pos_arg_ptr]
            pos_arg_ptr += 1
        if include_track_ids:
            track_ids = args[pos_arg_ptr]

        return visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            classes,
            scores,
            category_index=category_index,
            instance_masks=masks,
            keypoints=keypoints,
            track_ids=track_ids,
            **kwargs)

    return visualization_py_func_fn


def _resize_original_image(image, image_shape):
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_images(
        image,
        image_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        align_corners=True)
    return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         keypoint_edges=None,
                                         track_ids=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
    """Draws bounding boxes, masks, and keypoints on batch of image tensors.

    Args:
        images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
            channels will be ignored. If C = 1, then we convert the images to RGB
            images.
        boxes: [N, max_detections, 4] float32 tensor of detection boxes.
        classes: [N, max_detections] int tensor of detection classes. Note that
            classes are 1-indexed.
        scores: [N, max_detections] float32 tensor of detection scores.
        category_index: a dict that maps integer ids to category dicts. e.g.
            {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
        original_image_spatial_shape: [N, 2] tensor containing the spatial size of
            the original image.
        true_image_shape: [N, 3] tensor containing the spatial size of unpadded
            original_image.
        instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
            instance masks.
        keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
            with keypoints.
        keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.
        track_ids: [N, max_detections] int32 tensor of unique tracks ids (i.e.
            instance ids for each object). If provided, the color-coding of boxes is
            dictated by these ids, and not classes.
        max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
        min_score_thresh: Minimum score threshold for visualization. Default 0.2.
        use_normalized_coordinates: Whether to assume boxes and kepoints are in
            normalized coordinates (as opposed to absolute coordiantes). Default is
            True.

    Returns:
        4D image tensor of type uint8, with boxes drawn on top.
    """
    # Additional channels are being ignored.
    if images.shape[3] > 3:
        images = images[:, :, :, 0:3]
    elif images.shape[3] == 1:
        images = tf.image.grayscale_to_rgb(images)
    visualization_keyword_args = {
        'use_normalized_coordinates': use_normalized_coordinates,
        'max_boxes_to_draw': max_boxes_to_draw,
        'min_score_thresh': min_score_thresh,
        'agnostic_mode': False,
        'line_thickness': 4,
        'keypoint_edges': keypoint_edges
    }
    if true_image_shape is None:
        true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
    else:
        true_shapes = true_image_shape
    if original_image_spatial_shape is None:
        original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
    else:
        original_shapes = original_image_spatial_shape

    visualize_boxes_fn = create_visualization_fn(
        category_index,
        include_masks=instance_masks is not None,
        include_keypoints=keypoints is not None,
        include_track_ids=track_ids is not None,
        **visualization_keyword_args)

    elems = [true_shapes, original_shapes, images, boxes, classes, scores]
    if instance_masks is not None:
        elems.append(instance_masks)
    if keypoints is not None:
        elems.append(keypoints)
    if track_ids is not None:
        elems.append(track_ids)

    def draw_boxes(image_and_detections):
        """Draws boxes on image."""
        true_shape = image_and_detections[0]
        original_shape = image_and_detections[1]
        image = None
        if true_image_shape is not None:
            image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                               [true_shape[0], true_shape[1], 3])
        if original_image_spatial_shape is not None:
            image_and_detections[2] = _resize_original_image(image, original_shape)

        image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                      tf.uint8)
        return image_with_boxes

    images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
    return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True,
                                       keypoint_edges=None):
    """Creates a side-by-side image with detections and groundtruth.

    Bounding boxes (and instance masks, if available) are visualized on both
    subimages.

    Args:
        eval_dict: The evaluation dictionary returned by
            eval_util.result_dict_for_batched_example() or
            eval_util.result_dict_for_single_example().
        category_index: A category index (dictionary) produced from a labelmap.
        max_boxes_to_draw: The maximum number of boxes to draw for detections.
        min_score_thresh: The minimum score threshold for showing detections.
        use_normalized_coordinates: Whether to assume boxes and keypoints are in
            normalized coordinates (as opposed to absolute coordinates). Default is
            True.
        keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.

    Returns:
        A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
            corresponds to detections, while the subimage on the right corresponds to
            groundtruth.
    """
    detection_fields = fields.DetectionResultFields()
    input_data_fields = fields.InputDataFields()

    images_with_detections_list = []

    # Add the batch dimension if the eval_dict is for single example.
    if len(eval_dict[detection_fields.detection_classes].shape) == 1:
        for key in eval_dict:
            if key not in [input_data_fields.original_image, input_data_fields.image_additional_channels]:
                eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

    for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
        instance_masks = None
        if detection_fields.detection_masks in eval_dict:
            instance_masks = tf.cast(
                tf.expand_dims(
                    eval_dict[detection_fields.detection_masks][indx], axis=0),
                tf.uint8)
        keypoints = None
        if detection_fields.detection_keypoints in eval_dict:
            keypoints = tf.expand_dims(
                eval_dict[detection_fields.detection_keypoints][indx], axis=0)
        groundtruth_instance_masks = None
        if input_data_fields.groundtruth_instance_masks in eval_dict:
            groundtruth_instance_masks = tf.cast(
                tf.expand_dims(
                    eval_dict[input_data_fields.groundtruth_instance_masks][indx],
                    axis=0), tf.uint8)

        images_with_detections = draw_bounding_boxes_on_image_tensors(
            tf.expand_dims(
                eval_dict[input_data_fields.original_image][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_boxes][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_classes][indx], axis=0),
            tf.expand_dims(
                eval_dict[detection_fields.detection_scores][indx], axis=0),
            category_index,
            original_image_spatial_shape=tf.expand_dims(
                eval_dict[input_data_fields.original_image_spatial_shape][indx],
                axis=0),
            true_image_shape=tf.expand_dims(
                eval_dict[input_data_fields.true_image_shape][indx], axis=0),
            instance_masks=instance_masks,
            keypoints=keypoints,
            keypoint_edges=keypoint_edges,
            max_boxes_to_draw=max_boxes_to_draw,
            min_score_thresh=min_score_thresh,
            use_normalized_coordinates=use_normalized_coordinates)
        images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
            tf.expand_dims(
                eval_dict[input_data_fields.original_image][indx], axis=0),
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
            tf.expand_dims(
                eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
            tf.expand_dims(
                tf.ones_like(
                    eval_dict[input_data_fields.groundtruth_classes][indx],
                    dtype=tf.float32),
                axis=0),
            category_index,
            original_image_spatial_shape=tf.expand_dims(
                eval_dict[input_data_fields.original_image_spatial_shape][indx],
                axis=0),
            true_image_shape=tf.expand_dims(
                eval_dict[input_data_fields.true_image_shape][indx], axis=0),
            instance_masks=groundtruth_instance_masks,
            keypoints=None,
            keypoint_edges=None,
            max_boxes_to_draw=None,
            min_score_thresh=0.0,
            use_normalized_coordinates=use_normalized_coordinates)
        images_to_visualize = tf.concat(  # noqa pylint: disable=E1123
            [images_with_detections, images_with_groundtruth], axis=2)

        if input_data_fields.image_additional_channels in eval_dict:
            images_with_additional_channels_groundtruth = (
                draw_bounding_boxes_on_image_tensors(
                    tf.expand_dims(
                        eval_dict[input_data_fields.image_additional_channels][indx],
                        axis=0),
                    tf.expand_dims(
                        eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
                    tf.expand_dims(
                        eval_dict[input_data_fields.groundtruth_classes][indx],
                        axis=0),
                    tf.expand_dims(
                        tf.ones_like(
                            eval_dict[input_data_fields.groundtruth_classes][indx],
                            dtype=tf.float32),
                        axis=0),
                    category_index,
                    original_image_spatial_shape=tf.expand_dims(
                        eval_dict[input_data_fields.original_image_spatial_shape]
                        [indx],
                        axis=0),
                    true_image_shape=tf.expand_dims(
                        eval_dict[input_data_fields.true_image_shape][indx], axis=0),
                    instance_masks=groundtruth_instance_masks,
                    keypoints=None,
                    keypoint_edges=None,
                    max_boxes_to_draw=None,
                    min_score_thresh=0.0,
                    use_normalized_coordinates=use_normalized_coordinates))
            images_to_visualize = tf.concat(  # noqa pylint: disable=E1123
                [images_to_visualize, images_with_additional_channels_groundtruth],
                axis=2)
        images_with_detections_list.append(images_to_visualize)

    return images_with_detections_list


def draw_keypoints_on_image_array(
        image,
        keypoints,
        color='red',
        radius=2,
        use_normalized_coordinates=True,
        keypoint_edges=None,
        keypoint_edge_color='green',
        keypoint_edge_width=2):
    """Draws keypoints on an image (numpy array).

    Args:
        image: a numpy array with shape [height, width, 3].
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        radius: keypoint radius. Default value is 2.
        use_normalized_coordinates: if True (default), treat keypoint values as
            relative to the image.  Otherwise treat them as absolute.
        keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.
        keypoint_edge_color: color to draw the keypoint edges with. Default is red.
        keypoint_edge_width: width of the edges drawn between keypoints. Default
            value is 2.
    """
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    draw_keypoints_on_image(image_pil, keypoints, color, radius,
                            use_normalized_coordinates, keypoint_edges,
                            keypoint_edge_color, keypoint_edge_width)
    np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True,
                            keypoint_edges=None,
                            keypoint_edge_color='green',
                            keypoint_edge_width=2):
    """Draws keypoints on an image.

    Args:
        image: a PIL.Image object.
        keypoints: a numpy array with shape [num_keypoints, 2].
        color: color to draw the keypoints with. Default is red.
        radius: keypoint radius. Default value is 2.
        use_normalized_coordinates: if True (default), treat keypoint values as
            relative to the image.  Otherwise treat them as absolute.
        keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.
        keypoint_edge_color: color to draw the keypoint edges with. Default is red.
        keypoint_edge_width: width of the edges drawn between keypoints. Default
            value is 2.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    keypoints_x = [k[1] for k in keypoints]
    keypoints_y = [k[0] for k in keypoints]
    if use_normalized_coordinates:
        keypoints_x = [im_width * x for x in keypoints_x]  # removed tuple(*)
        keypoints_y = [im_height * y for y in keypoints_y]  # removed tuple(*)
    for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
        draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                      (keypoint_x + radius, keypoint_y + radius)],
                     outline=color,
                     fill=color)
    if keypoint_edges is not None:
        for keypoint_start, keypoint_end in keypoint_edges:
            if (keypoint_start < 0 or keypoint_start >= len(keypoints) or
                    keypoint_end < 0 or keypoint_end >= len(keypoints)):
                continue
            edge_coordinates = [
                keypoints_x[keypoint_start], keypoints_y[keypoint_start],
                keypoints_x[keypoint_end], keypoints_y[keypoint_end]
            ]
            draw.line(
                edge_coordinates, fill=keypoint_edge_color, width=keypoint_edge_width)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
    """Draws mask on an image.

    Args:
        image: uint8 numpy array with shape (img_height, img_height, 3)
        mask: a uint8 numpy array of shape (img_height, img_height) with values
            between either 0 or 1.
        color: color to draw the keypoints with. Default is red.
        alpha: transparency value between 0 and 1. (default: 0.4)

    Raises:
        ValueError: On incorrect data type for image or masks.
    """
    if image.dtype != np.uint8:
        raise ValueError('`image` not of type np.uint8')
    if mask.dtype != np.uint8:
        raise ValueError('`mask` not of type np.uint8')
    if np.any(np.logical_and(mask != 1, mask != 0)):
        raise ValueError('`mask` elements should be in [0, 1]')
    if image.shape[:2] != mask.shape:
        raise ValueError(f'The image has spatial dimensions {image.shape[:2]} but the mask has '
                         'dimensions {mask.shape}')
    rgb = ImageColor.getrgb(color)
    pil_image = Image.fromarray(image)

    solid_color = np.expand_dims(
        np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
    pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
    pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
    np.copyto(image, np.array(pil_image.convert('RGB')))


def visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index,
        instance_masks=None,
        instance_boundaries=None,
        keypoints=None,
        keypoint_edges=None,
        track_ids=None,
        use_normalized_coordinates=False,
        max_boxes_to_draw=20,
        min_score_thresh=.5,
        agnostic_mode=False,
        line_thickness=4,
        groundtruth_box_visualization_color='black',
        skip_boxes=False,
        skip_scores=False,
        skip_labels=False,
        skip_track_ids=False):
    """Overlay labeled boxes on an image with formatted scores and label names.

    This function groups boxes that correspond to the same location
    and creates a display string for each detection and overlays these
    on the image. Note that this function modifies the image in place, and returns
    that same image.

    Args:
        image: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then this
            function assumes that the boxes to be plotted are groundtruth boxes and
            plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
        instance_masks: a numpy array of shape [N, image_height, image_width] with
            values ranging between 0 and 1, can be None.
        instance_boundaries: a numpy array of shape [N, image_height, image_width]
            with values ranging between 0 and 1, can be None.
        keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
        keypoint_edges: A list of tuples with keypoint indices that specify which
            keypoints should be connected by an edge, e.g. [(0, 1), (2, 4)] draws
            edges from keypoint 0 to 1 and from keypoint 2 to 4.
        track_ids: a numpy array of shape [N] with unique track ids. If provided,
            color-coding of boxes will be determined by these ids, and not the class
            indices.
        use_normalized_coordinates: whether boxes is to be interpreted as normalized
            coordinates or not.
        max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw all
            boxes.
        min_score_thresh: minimum score threshold for a box to be visualized
        agnostic_mode: boolean (default: False) controlling whether to evaluate in
            class-agnostic mode or not.  This mode will display scores but ignore
            classes.
        line_thickness: integer (default: 4) controlling line width of the boxes.
        groundtruth_box_visualization_color: box color for visualizing groundtruth
            boxes
        skip_boxes: whether to skip the drawing of bounding boxes.
        skip_scores: whether to skip score when drawing a single detection
        skip_labels: whether to skip label when drawing a single detection
        skip_track_ids: whether to skip track id when drawing a single detection

    Returns:
        uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
    """
    # Create a display string (and color) for every box location, group any boxes
    # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_instance_boundaries_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    box_to_track_ids_map = {}
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(boxes.shape[0]):
        if max_boxes_to_draw == len(box_to_color_map):
            break
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if instance_masks is not None:
                box_to_instance_masks_map[box] = instance_masks[i]
            if instance_boundaries is not None:
                box_to_instance_boundaries_map[box] = instance_boundaries[i]
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if track_ids is not None:
                box_to_track_ids_map[box] = track_ids[i]
            if scores is None:
                box_to_color_map[box] = groundtruth_box_visualization_color
            else:
                display_str = ''
                if not skip_labels:
                    if not agnostic_mode:
                        if classes[i] in six.viewkeys(category_index):
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        display_str = str(class_name)
                if not skip_scores:
                    if not display_str:
                        display_str = f'{int(100 * scores[i])}%'
                    else:
                        display_str = f'{display_str}: {int(100 * scores[i])}%'
                if not skip_track_ids and track_ids is not None:
                    if not display_str:
                        display_str = f'ID {track_ids[i]}'
                    else:
                        display_str = f'{display_str}: ID {track_ids[i]}'
                box_to_display_str_map[box].append(display_str)
                if agnostic_mode:
                    box_to_color_map[box] = 'DarkOrange'
                elif track_ids is not None:
                    prime_multipler = _get_multiplier_for_color_randomness()
                    box_to_color_map[box] = STANDARD_COLORS[(prime_multipler * track_ids[i]) %
                                                            len(STANDARD_COLORS)]
                else:
                    box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

    # Draw all boxes onto image.
    for box, color in box_to_color_map.items():
        ymin, xmin, ymax, xmax = box
        if instance_masks is not None:
            draw_mask_on_image_array(
                image, box_to_instance_masks_map[box], color=color)
        if instance_boundaries is not None:
            draw_mask_on_image_array(
                image, box_to_instance_boundaries_map[box], color='red', alpha=1.0)
        draw_bounding_box_on_image_array(
            image,
            ymin,
            xmin,
            ymax,
            xmax,
            color=color,
            thickness=0 if skip_boxes else line_thickness,
            display_str_list=box_to_display_str_map[box],
            use_normalized_coordinates=use_normalized_coordinates)
        if keypoints is not None:
            draw_keypoints_on_image_array(
                image,
                box_to_keypoints_map[box],
                color=color,
                radius=line_thickness / 2,
                use_normalized_coordinates=use_normalized_coordinates,
                keypoint_edges=keypoint_edges,
                keypoint_edge_color=color,
                keypoint_edge_width=line_thickness // 2)

    return image


def add_cdf_image_summary(values, name):
    """Adds a tf.summary.image for a CDF plot of the values.

    Normalizes `values` such that they sum to 1, plots the cumulative distribution
    function and creates a tf image summary.

    Args:
        values: a 1-D float32 tensor containing the values.
        name: name for the image summary.
    """

    def cdf_plot(values):
        """Numpy function to plot CDF."""
        normalized_values = values / np.sum(values)
        sorted_values = np.sort(normalized_values)
        cumulative_values = np.cumsum(sorted_values)
        fraction_of_examples = (
            np.arange(cumulative_values.size, dtype=np.float32) /
            cumulative_values.size)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        ax.plot(fraction_of_examples, cumulative_values)
        ax.set_ylabel('cumulative normalized values')
        ax.set_xlabel('fraction of examples')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(1, int(height), int(width), 3)
        return image

    cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
    tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
    """Adds a tf.summary.image for a histogram plot of the values.

    Plots the histogram of values and creates a tf image summary.

    Args:
        values: a 1-D float32 tensor containing the values.
        bins: bin edges which will be directly passed to np.histogram.
        name: name for the image summary.
    """

    def hist_plot(values, bins):
        """Numpy function to plot hist."""
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot('111')
        y, x = np.histogram(values, bins=bins)
        ax.plot(x[:-1], y)
        ax.set_ylabel('count')
        ax.set_xlabel('value')
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(1, int(height), int(width), 3)
        return image

    hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
    tf.summary.image(name, hist_plot)


def denormalize_image(image):
    """De-normalize image array.

    Args:
        image: numpy array [H, W, C]
    Return:
        output: numpy array [H, W, C]
    """
    scale = np.array([0.224, 0.224, 0.224])[None][None]
    mean = np.array([0.485, 0.456, 0.406])[None][None]
    output = (image * scale + mean) * 255
    return output.astype(np.uint8)


def visualize_detections(image_path, output_path, detections, labels):
    """Visualize detections."""
    image = Image.open(image_path).convert(mode='RGB')
    draw = ImageDraw.Draw(image)
    line_width = 2
    font = ImageFont.load_default()
    for d in detections:
        color = STANDARD_COLORS[d['class'] % len(STANDARD_COLORS)]
        draw.line([(d['xmin'], d['ymin']), (d['xmin'], d['ymax']),
                   (d['xmax'], d['ymax']), (d['xmax'], d['ymin']),
                   (d['xmin'], d['ymin'])], width=line_width, fill=color)
        label = f"Class {d['class']}"
        if d['class'] < len(labels):
            label = str(labels[d['class']])
        score = d['score']
        text = f"{label}: {int(100 * score)}%"
        if score < 0:
            text = label
        text_width = font.getbbox(text)[2] - font.getbbox(text)[0]
        text_height = font.getbbox(text)[3] - font.getbbox(text)[1]
        text_bottom = max(text_height, d['ymin'])
        text_left = d['xmin']
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(text_left, text_bottom - text_height - 2 * margin), (text_left + text_width, text_bottom)],
            fill=color)
        draw.text(
            (text_left + margin, text_bottom - text_height - margin),
            text, fill='black', font=font)
    image.save(output_path)
