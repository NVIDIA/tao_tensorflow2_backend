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
# ==============================================================================
"""Data loader and processing."""
import logging
import os
import multiprocessing
import tensorflow as tf

from nvidia_tao_tf2.blocks.dataloader.dataset import Dataset
from nvidia_tao_tf2.cv.efficientdet.utils import model_utils
from nvidia_tao_tf2.cv.efficientdet.model import anchors
from nvidia_tao_tf2.cv.efficientdet.utils.horovod_utils import get_rank, get_world_size, is_main_process
from nvidia_tao_tf2.cv.efficientdet.utils.keras_utils import get_mixed_precision_policy

from nvidia_tao_tf2.cv.core import preprocessor
from nvidia_tao_tf2.cv.core import tf_example_decoder
logger = logging.getLogger(__name__)


class InputProcessor:
    """Base class of Input processor."""

    def __init__(self, image, output_size):
        """Initializes a new `InputProcessor`.

        Args:
            image: The input image before processing.
            output_size: The output image size after calling resize_and_crop_image
                function.
        """
        self._image = image
        if isinstance(output_size, int):
            self._output_size = (output_size, output_size)
        else:
            self._output_size = output_size
        # Parameters to control rescaling and shifting during preprocessing.
        # Image scale defines scale from original image to scaled image.
        self._image_scale = tf.constant(1.0)
        # The integer height and width of scaled image.
        self._scaled_height = tf.shape(image)[0]
        self._scaled_width = tf.shape(image)[1]
        # The x and y translation offset to crop scaled image to the output size.
        self._crop_offset_y = tf.constant(0)
        self._crop_offset_x = tf.constant(0)

    def normalize_image(self, dtype=tf.float32):
        """Normalize the image to zero mean and unit variance."""
        self._image = tf.image.convert_image_dtype(self._image, dtype=dtype)
        offset = tf.constant([0.485, 0.456, 0.406])
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        self._image -= offset

        scale = tf.constant([0.224, 0.224, 0.224])
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        self._image /= scale

    def get_image(self):
        """Get image."""
        return self._image

    def set_training_random_scale_factors(self,
                                          scale_min,
                                          scale_max,
                                          target_size=None):
        """Set the parameters for multiscale training.

        Notably, if train and eval use different sizes, then target_size should be
        set as eval size to avoid the discrency between train and eval.

        Args:
            scale_min: minimal scale factor.
            scale_max: maximum scale factor.
            target_size: targeted size, usually same as eval. If None, use train size.
        """
        if not target_size:
            target_size = self._output_size
        target_size = model_utils.parse_image_size(target_size)
        if is_main_process():
            logger.debug('target_size = %s, output_size = %s', target_size, self._output_size)

        # Select a random scale factor.
        random_scale_factor = tf.random.uniform([], scale_min, scale_max)
        scaled_y = tf.cast(random_scale_factor * target_size[0], tf.int32)
        scaled_x = tf.cast(random_scale_factor * target_size[1], tf.int32)

        # Recompute the accurate scale_factor using rounded scaled image size.
        height = tf.cast(tf.shape(self._image)[0], tf.float32)
        width = tf.cast(tf.shape(self._image)[1], tf.float32)
        image_scale_y = tf.cast(scaled_y, tf.float32) / height
        image_scale_x = tf.cast(scaled_x, tf.float32) / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)

        # Select non-zero random offset (x, y) if scaled image is larger than
        # self._output_size.
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        offset_y = tf.cast(scaled_height - self._output_size[0], tf.float32)
        offset_x = tf.cast(scaled_width - self._output_size[1], tf.float32)
        offset_y = tf.maximum(0.0, offset_y) * tf.random.uniform([], 0, 1)
        offset_x = tf.maximum(0.0, offset_x) * tf.random.uniform([], 0, 1)
        offset_y = tf.cast(offset_y, tf.int32)
        offset_x = tf.cast(offset_x, tf.int32)
        self._image_scale = image_scale
        self._scaled_height = scaled_height
        self._scaled_width = scaled_width
        self._crop_offset_x = offset_x
        self._crop_offset_y = offset_y

    def set_scale_factors_to_output_size(self):
        """Set the parameters to resize input image to self._output_size."""
        # Compute the scale_factor using rounded scaled image size.
        height = tf.cast(tf.shape(self._image)[0], tf.float32)
        width = tf.cast(tf.shape(self._image)[1], tf.float32)
        image_scale_y = tf.cast(self._output_size[0], tf.float32) / height
        image_scale_x = tf.cast(self._output_size[1], tf.float32) / width
        image_scale = tf.minimum(image_scale_x, image_scale_y)
        scaled_height = tf.cast(height * image_scale, tf.int32)
        scaled_width = tf.cast(width * image_scale, tf.int32)
        self._image_scale = image_scale
        self._scaled_height = scaled_height
        self._scaled_width = scaled_width

    def resize_and_crop_image(self, method=tf.image.ResizeMethod.BILINEAR):
        """Resize input image and crop it to the self._output dimension."""
        # dtype = self._image.dtype
        scaled_image = tf.compat.v1.image.resize(
            self._image, [self._scaled_height, self._scaled_width], method=method)
        scaled_image = scaled_image[self._crop_offset_y:self._crop_offset_y +
                                    self._output_size[0],
                                    self._crop_offset_x:self._crop_offset_x +
                                    self._output_size[1], :]
        self._image = tf.image.pad_to_bounding_box(scaled_image, 0, 0,
                                                   self._output_size[0],
                                                   self._output_size[1])
        # self._image = tf.cast(output_image, dtype) # TODO(@yuw): verify
        # self._image = tf.transpose(self._image, [2, 0, 1])
        return self._image


class DetectionInputProcessor(InputProcessor):
    """Input processor for object detection."""

    def __init__(self, image, output_size, boxes=None, classes=None):
        """Init."""
        InputProcessor.__init__(self, image, output_size)
        self._boxes = boxes
        self._classes = classes

    def random_horizontal_flip(self):
        """Randomly flip input image and bounding boxes."""
        self._image, self._boxes = preprocessor.random_horizontal_flip(
            self._image, boxes=self._boxes)

    def clip_boxes(self, boxes):
        """Clip boxes to fit in an image."""
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        ymin = tf.clip_by_value(ymin, 0, self._output_size[0] - 1)
        xmin = tf.clip_by_value(xmin, 0, self._output_size[1] - 1)
        ymax = tf.clip_by_value(ymax, 0, self._output_size[0] - 1)
        xmax = tf.clip_by_value(xmax, 0, self._output_size[1] - 1)
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
        return boxes

    def resize_and_crop_boxes(self):
        """Resize boxes and crop it to the self._output dimension."""
        boxlist = preprocessor.box_list.BoxList(self._boxes)
        # boxlist is in range of [0, 1], so here we pass the scale_height/width
        # instead of just scale.
        boxes = preprocessor.box_list_scale(boxlist, self._scaled_height, self._scaled_width).get()
        # Adjust box coordinates based on the offset.
        box_offset = tf.stack([
            self._crop_offset_y,
            self._crop_offset_x,
            self._crop_offset_y,
            self._crop_offset_x,
        ])
        boxes -= tf.cast(tf.reshape(box_offset, [1, 4]), tf.float32)
        # Clip the boxes.
        boxes = self.clip_boxes(boxes)
        # Filter out ground truth boxes that are illegal.
        indices = tf.where(
            tf.not_equal((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 0))
        boxes = tf.gather_nd(boxes, indices)
        classes = tf.gather_nd(self._classes, indices)
        return boxes, classes

    @property
    def image_scale(self):
        """Return image scale from original image to scaled image."""
        return self._image_scale

    @property
    def image_scale_to_original(self):
        """Return image scale from scaled image to original image."""
        return 1.0 / self._image_scale

    @property
    def offset_x(self):
        """Return offset of x."""
        return self._crop_offset_x

    @property
    def offset_y(self):
        """Return offset of y."""
        return self._crop_offset_y


def pad_to_fixed_size(data, pad_value, output_shape):
    """Pad data to a fixed length at the first dimension.

    Args:
        data: Tensor to be padded to output_shape.
        pad_value: A constant value assigned to the paddings.
        output_shape: The output shape of a 2D tensor.
    Returns:
        The Padded tensor with output_shape [max_instances_per_image, dimension].
    """
    max_instances_per_image = output_shape[0]
    dimension = output_shape[1]
    data = tf.reshape(data, [-1, dimension])
    num_instances = tf.shape(data)[0]
    msg = 'ERROR: please increase config.max_instances_per_image'
    with tf.control_dependencies(
            [tf.assert_less(num_instances, max_instances_per_image, message=msg)]):
        pad_length = max_instances_per_image - num_instances
    paddings = pad_value * tf.ones([pad_length, dimension])
    padded_data = tf.concat([data, paddings], axis=0)  # noqa pylint: disable=E1123
    padded_data = tf.reshape(padded_data, output_shape)
    return padded_data


class CocoDataset(Dataset):
    """COCO Dataset."""

    def __init__(self,
                 data_sources,
                 is_training,
                 use_fake_data=False,
                 max_instances_per_image=None,
                 sampling='uniform'):
        """Dataloader for COCO format dataset.

        Args:
            data_sources (DataSource): data source for train/val set.
            is_training (bool): training phase.
            use_fake_data (bool, optional): Whether to use fake data. Defaults to False.
            max_instances_per_image (int, optional): Max number of instances to detect per image.
            sampling (str, optional): sampling method. Defaults to 'uniform'.
        """
        self._data_sources = data_sources
        self._is_training = is_training
        self._use_fake_data = use_fake_data
        # COCO has 100 limit, but users may set different values for custom dataset.
        self._max_instances_per_image = max_instances_per_image or 100
        assert sampling in ['uniform', 'proportional'], \
            f"Sampling method {sampling} is not supported."
        self._sampling = sampling

    @tf.autograph.experimental.do_not_convert
    def dataset_parser(self, value, example_decoder, anchor_labeler, params):
        """Parse data to a fixed dimension input image and learning targets.

        Args:
            value: a single serialized tf.Example string.
            example_decoder: TF example decoder.
            anchor_labeler: anchor box labeler.
            params: a dict of extra parameters.

        Returns:
            image: Image tensor that is preprocessed to have normalized value and
                fixed dimension [image_height, image_width, 3]
            cls_targets_dict: ordered dictionary with keys
                [min_level, min_level+1, ..., max_level]. The values are tensor with
                shape [height_l, width_l, num_anchors]. The height_l and width_l
                represent the dimension of class logits at l-th level.
            box_targets_dict: ordered dictionary with keys
                [min_level, min_level+1, ..., max_level]. The values are tensor with
                shape [height_l, width_l, num_anchors * 4]. The height_l and
                width_l represent the dimension of bounding box regression output at
                l-th level.
            num_positives: Number of positive anchors in the image.
            source_id: Source image id. Default value -1 if the source id is empty
                in the groundtruth annotation.
            image_scale: Scale of the processed image to the original image.
            boxes: Groundtruth bounding box annotations. The box is represented in
                [y1, x1, y2, x2] format. The tensor is padded with -1 to the fixed
                dimension [self._max_instances_per_image, 4].
            is_crowds: Groundtruth annotations to indicate if an annotation
                represents a group of instances by value {0, 1}. The tensor is
                padded with 0 to the fixed dimension [self._max_instances_per_image].
            areas: Groundtruth areas annotations. The tensor is padded with -1
                to the fixed dimension [self._max_instances_per_image].
            classes: Groundtruth classes annotations. The tensor is padded with -1
                to the fixed dimension [self._max_instances_per_image].
        """
        with tf.name_scope('parser'):
            data = example_decoder.decode(value)
            source_id = data['source_id']
            image = data['image']
            boxes = data['groundtruth_boxes']
            classes = data['groundtruth_classes']
            classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])
            areas = data['groundtruth_area']
            is_crowds = data['groundtruth_is_crowd']
            image_masks = data.get('groundtruth_instance_masks', [])
            classes = tf.reshape(tf.cast(classes, dtype=tf.float32), [-1, 1])

            if self._is_training:
                # Training time preprocessing.
                if params['skip_crowd_during_training']:
                    indices = tf.where(tf.logical_not(data['groundtruth_is_crowd']))
                    classes = tf.gather_nd(classes, indices)
                    boxes = tf.gather_nd(boxes, indices)

                if params.get('auto_augment', None):
                    from nvidia_tao_tf2.cv.efficientdet.augmentation import autoaugment  # noqa pylint: disable=C0415
                    if params['auto_color']:
                        if is_main_process():
                            logger.info("Auto color augmentation is enabled.")
                        image, boxes = autoaugment.distort_image_with_autocolor(
                            image, boxes, num_layers=1, magnitude=15)
                    if params['auto_translate_xy']:
                        if is_main_process():
                            logger.info("Auto translate_xy augmentation is enabled.")
                        image, boxes = autoaugment.distort_image_with_autotranslate(
                            image, boxes, num_layers=1, magnitude=15)

            input_processor = DetectionInputProcessor(image, params['image_size'], boxes, classes)
            input_processor.normalize_image()

            if self._is_training:
                if params['input_rand_hflip']:
                    input_processor.random_horizontal_flip()

                input_processor.set_training_random_scale_factors(
                    params['jitter_min'], params['jitter_max'],
                    params.get('target_size', None))
            else:
                input_processor.set_scale_factors_to_output_size()
            image = input_processor.resize_and_crop_image()
            boxes, classes = input_processor.resize_and_crop_boxes()

            # Assign anchors.
            (cls_targets, box_targets,
             num_positives) = anchor_labeler.label_anchors(boxes, classes)

            source_id = tf.where(
                tf.equal(source_id, tf.constant('')), '-1', source_id)
            source_id = tf.strings.to_number(source_id)

            # Pad groundtruth data for evaluation.
            image_scale = input_processor.image_scale_to_original
            boxes *= image_scale
            is_crowds = tf.cast(is_crowds, dtype=tf.float32)
            boxes = pad_to_fixed_size(boxes, -1, [self._max_instances_per_image, 4])
            is_crowds = pad_to_fixed_size(is_crowds, 0, [self._max_instances_per_image, 1])
            areas = pad_to_fixed_size(areas, -1, [self._max_instances_per_image, 1])
            classes = pad_to_fixed_size(classes, -1, [self._max_instances_per_image, 1])
            if params['mixed_precision']:
                dtype = get_mixed_precision_policy().compute_dtype
                image = tf.cast(image, dtype=dtype)
                box_targets = tf.nest.map_structure(
                    lambda box_target: tf.cast(box_target, dtype=tf.float32), box_targets)
            return (image, cls_targets, box_targets, num_positives, source_id,
                    image_scale, boxes, is_crowds, areas, classes, image_masks)

    @tf.autograph.experimental.do_not_convert
    def process_example(self, params, batch_size, images, cls_targets,
                        box_targets, num_positives, source_ids, image_scales,
                        boxes, is_crowds, areas, classes, image_masks):
        """Processes one batch of data."""
        labels = {}
        # Count num_positives in a batch.
        num_positives_batch = tf.reduce_mean(num_positives)
        labels['mean_num_positives'] = tf.reshape(
            tf.tile(tf.expand_dims(num_positives_batch, 0), [
                batch_size,
            ]), [batch_size, 1])

        if params['data_format'] == 'channels_first':
            images = tf.transpose(images, [0, 3, 1, 2])

        for level in range(params['min_level'], params['max_level'] + 1):
            labels[f'cls_targets_{level}'] = cls_targets[level]
            labels[f'box_targets_{level}'] = box_targets[level]
            if params['data_format'] == 'channels_first':
                labels[f'cls_targets_{level}'] = tf.transpose(
                    labels[f'cls_targets_{level}'], [0, 3, 1, 2])
                labels[f'box_targets_{level}'] = tf.transpose(
                    labels[f'box_targets_{level}'], [0, 3, 1, 2])
        # Concatenate groundtruth annotations to a tensor.
        groundtruth_data = tf.concat([boxes, is_crowds, areas, classes], axis=2)  # noqa pylint: disable=E1123
        labels['source_ids'] = source_ids
        labels['groundtruth_data'] = groundtruth_data
        labels['image_scales'] = image_scales
        labels['image_masks'] = image_masks
        return images, labels

    @property
    def dataset_options(self):
        """Dataset options."""
        options = tf.data.Options()
        options.experimental_deterministic = not self._is_training
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        return options

    def __call__(self, params, batch_size=None):
        """Call."""
        input_anchors = anchors.Anchors(params['min_level'], params['max_level'],
                                        params['num_scales'],
                                        params['aspect_ratios'],
                                        params['anchor_scale'],
                                        params['image_size'])
        anchor_labeler = anchors.AnchorLabeler(input_anchors, params['num_classes'])
        example_decoder = tf_example_decoder.TfExampleDecoder(
            include_mask='segmentation' in params['heads'],
            regenerate_source_id=params['regenerate_source_id'],
            include_image=True,  # TODO(@yuw): forced to True, as image must be encoded in tfrecords
        )

        batch_size = batch_size or params['batch_size']
        datasets = []
        weights = []
        for file_pattern, image_dir in self._data_sources:
            dataset = tf.data.Dataset.list_files(
                file_pattern,
                shuffle=params['shuffle_file'] and self._is_training)
            if self._is_training:
                dataset = dataset.shard(get_world_size(), get_rank())
                dataset.shuffle(buffer_size=64)

            # Prefetch data from files.
            def _prefetch_dataset(filename):
                dataset = []
                if params.get('dataset_type', None) == 'sstable':
                    pass
                else:
                    dataset = tf.data.TFRecordDataset(filename).prefetch(1)
                return dataset

            dataset = dataset.interleave(
                _prefetch_dataset, cycle_length=params['cycle_length'],  # cycle_length=32
                block_length=params['block_length'],  # block_length=16
                num_parallel_calls=tf.data.experimental.AUTOTUNE)  # TODO(@yuw): whether to fix the number

            dataset = dataset.with_options(self.dataset_options)
            if self._is_training:
                dataset = dataset.shuffle(
                    buffer_size=params['shuffle_buffer'],
                    reshuffle_each_iteration=True,)
            # Parse the fetched records to input tensors for model function.
            # pylint: disable=g-long-lambda
            map_fn = lambda value: self.dataset_parser(value, example_decoder,  # noqa pylint: disable=C3001
                                                       anchor_labeler, params)
            # pylint: enable=g-long-lambda
            # TODO(@yuw): whether to make it configurable or tf.data.experimental.AUTOTUNE
            core_count = multiprocessing.cpu_count() // 2  # note: // 2 to be conservative
            dataset = dataset.map(
                map_fn, num_parallel_calls=max(core_count // get_world_size(), 1))
            # dataset = dataset.prefetch(batch_size)
            if self._sampling and len(self._data_sources) > 1:
                dataset = dataset.repeat()
            datasets.append(dataset)
            weights.append(1 if (not image_dir or image_dir == '/') else len(os.listdir(image_dir)))

        if not self._sampling or len(self._data_sources) == 1 or not self._is_training:
            combined = datasets[0]
            for dataset in datasets[1:]:
                combined = combined.concatenate(dataset)
        elif self._sampling == 'proportional':
            if is_main_process():
                logger.info("Use Proportional Sampling")
            total = sum(weights)
            weights = [w / total for w in weights]
            combined = tf.data.Dataset.sample_from_datasets(
                datasets, weights=weights
            )
        elif self._sampling == 'uniform':
            if is_main_process():
                logger.info("Use Uniform Sampling")
            weights = [1.0 / len(weights)] * len(weights)
            combined = tf.data.Dataset.sample_from_datasets(
                datasets, weights=weights, stop_on_empty_dataset=True,
            )
        else:
            raise ValueError("Unexpected error in dataloader!")

        combined = combined.batch(batch_size, drop_remainder=params['drop_remainder'])
        combined = combined.map(
            lambda *args: self.process_example(params, batch_size, *args))
        combined = combined.prefetch(params['prefetch_size'] or tf.data.experimental.AUTOTUNE)
        if self._is_training:
            combined = combined.repeat()
        if self._use_fake_data:
            # Turn this dataset into a semi-fake dataset which always loop at the
            # first batch. This reduces variance in performance and is useful in
            # testing.
            combined = combined.take(1).cache().repeat()
        return combined
