# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Convert raw COCO dataset to TFRecord for object_detection."""
import collections
from collections import Counter
import hashlib
import io
import json
import multiprocessing
import os
import numpy as np
import PIL.Image
from pycocotools import mask
from skimage import measure

import tensorflow as tf

from common.dataset import dataset_util
from common.dataset import label_map_util
from common.decorators import monitor_status
from common.hydra.hydra_runner import hydra_runner
import common.logging.logging as status_logging

from cv.efficientdet.config.default_config import ExperimentConfig


def setup_env(cfg):
    """Setup data conversion env."""
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir)
    if not os.path.exists(cfg.dataset_convert.output_dir):
        os.mkdir(cfg.dataset_convert.output_dir)


def create_tf_example(image,
                      bbox_annotations,
                      image_dir,
                      category_index,
                      include_masks=False,
                      inspect_mask=True):
    """Converts image and annotations to a tf.Example proto.

    Args:
        image: dict with keys:
        [u'license', u'file_name', u'coco_url', u'height', u'width',
        u'date_captured', u'flickr_url', u'id']
        bbox_annotations:
        list of dicts with keys:
        [u'segmentation', u'area', u'iscrowd', u'image_id',
        u'bbox', u'category_id', u'id']
        Notice that bounding box coordinates in the official COCO dataset are
        given as [x, y, width, height] tuples using absolute coordinates where
        x, y represent the top-left (0-indexed) corner.  This function converts
        to the format expected by the Tensorflow Object Detection API (which is
        which is [ymin, xmin, ymax, xmax] with coordinates normalized relative
        to image size).
        image_dir: directory containing the image files.
        category_index: a dict containing COCO category information keyed
        by the 'id' field of each category.  See the
        label_map_util.create_category_index function.
        include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
    Returns:
        example: The converted tf.Example
        num_annotations_skipped: Number of (invalid) annotations that were ignored.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    image_height = image['height']
    image_width = image['width']
    filename = image['file_name']
    image_id = image['id']

    full_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    encoded_mask_png = []
    num_annotations_skipped = 0
    log_warnings = {}
    cat_counter = Counter()
    box_oob = []  # out of bound boxes
    mask_oob = []  # out of bound masks
    for object_annotations in bbox_annotations:
        object_annotations_id = object_annotations['id']
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:  # or x + width > image_width or y + height > image_height
            num_annotations_skipped += 1
            box_oob.append(object_annotations_id)
            continue

        # correct label errors
        left = max(x, 0)
        top = max(y, 0)
        right = min(left + width, image_width)
        bottom = min(top + height, image_height)
        if right - left < 1 or bottom - top < 1:
            num_annotations_skipped += 1
            box_oob.append(object_annotations_id)
            continue

        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_id = int(object_annotations['category_id'])
        category_ids.append(category_id)
        category_names.append(category_index[category_id]['name'].encode('utf8'))
        if str(category_index[category_id]['name']) in cat_counter:
            cat_counter[str(category_index[category_id]['name'])] += 1
        else:
            cat_counter[str(category_index[category_id]['name'])] = 1
        area.append(object_annotations['area'])

        if include_masks:
            if 'segmentation' not in object_annotations:
                raise ValueError(
                    f"segmentation groundtruth is missing in object: {object_annotations_id}.")
            # pylygon (e.g. [[289.74,443.39,302.29,445.32, ...], [1,2,3,4]])
            if isinstance(object_annotations['segmentation'], list):
                rles = mask.frPyObjects(object_annotations['segmentation'],
                                        image_height, image_width)
                rle = mask.merge(rles)
            elif 'counts' in object_annotations['segmentation']:
                # e.g. {'counts': [6, 1, 40, 4, 5, 4, 5, 4, 21], 'size': [9, 10]}
                if isinstance(object_annotations['segmentation']['counts'], list):
                    rle = mask.frPyObjects(object_annotations['segmentation'],
                                           image_height, image_width)
                else:
                    rle = object_annotations['segmentation']
            else:
                raise ValueError('Please check the segmentation format.')
            binary_mask = mask.decode(rle)
            contours = measure.find_contours(binary_mask, 0.5)
            if inspect_mask:
                # check if mask is out of bound compared to bbox
                min_x, max_x = image_width + 1, -1
                min_y, max_y = image_height + 1, -1
                for cont in contours:
                    c = np.array(cont)
                    min_x = min(min_x, np.amin(c, axis=0)[1])
                    max_x = max(max_x, np.amax(c, axis=0)[1])
                    min_y = min(min_y, np.amin(c, axis=0)[0])
                    max_y = max(max_y, np.amax(c, axis=0)[0])
                xxmin, xxmax, yymin, yymax = \
                    float(x) - 1, float(x + width) + 1, float(y) - 1, float(y + height) + 1
                if xxmin > min_x or yymin > min_y or xxmax < max_x or yymax < max_y:
                    mask_oob.append(object_annotations_id)

            # if not object_annotations['iscrowd']:
            #     binary_mask = np.amax(binary_mask, axis=2)
            pil_image = PIL.Image.fromarray(binary_mask)
            output_io = io.BytesIO()
            pil_image.save(output_io, format='PNG')
            encoded_mask_png.append(output_io.getvalue())

    feature_dict = {
        'image/height':
            dataset_util.int64_feature(image_height),
        'image/width':
            dataset_util.int64_feature(image_width),
        'image/filename':
            dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id':
            dataset_util.bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded':
            dataset_util.bytes_feature(encoded_jpg),
        'image/format':
            dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax':
            dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin':
            dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax':
            dataset_util.float_list_feature(ymax),
        'image/object/class/text':
            dataset_util.bytes_list_feature(category_names),
        'image/object/class/label':
            dataset_util.int64_list_feature(category_ids),
        'image/object/is_crowd':
            dataset_util.int64_list_feature(is_crowd),
        'image/object/area':
            dataset_util.float_list_feature(area),
    }
    if include_masks:
        feature_dict['image/object/mask'] = (
            dataset_util.bytes_list_feature(encoded_mask_png))
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    if mask_oob or box_oob:
        log_warnings[image_id] = {}
        log_warnings[image_id]['box'] = box_oob
        log_warnings[image_id]['mask'] = mask_oob
    return key, example, num_annotations_skipped, log_warnings, cat_counter


def _pool_create_tf_example(args):
    return create_tf_example(*args)


def _load_object_annotations(object_annotations_file):
    with tf.io.gfile.GFile(object_annotations_file, 'r') as fid:
        obj_annotations = json.load(fid)

    images = obj_annotations['images']
    category_index = label_map_util.create_category_index(
        obj_annotations['categories'])

    img_to_obj_annotation = collections.defaultdict(list)
    tf.compat.v1.logging.info('Building bounding box index.')
    for annotation in obj_annotations['annotations']:
        image_id = annotation['image_id']
        img_to_obj_annotation[image_id].append(annotation)

    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in img_to_obj_annotation:
            missing_annotation_count += 1

    tf.compat.v1.logging.info('%d images are missing bboxes.', missing_annotation_count)

    return images, img_to_obj_annotation, category_index


def _merge_log(log_a, log_b):
    log_ab = log_a.copy()
    for k, v in log_b.items():
        if k in log_ab:
            log_ab[k] += v
        else:
            log_ab[k] = v
    return log_ab


def _create_tf_record_from_coco_annotations(object_annotations_file,
                                            image_dir, output_path, include_masks, num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.

    Args:
        object_annotations_file: JSON file containing bounding box annotations.
        image_dir: Directory containing the image files.
        output_path: Path to output tf.Record file.
        include_masks: Whether to include instance segmentations masks
        (PNG encoded) in the result. default: False.
        num_shards: Number of output files to create.
    """
    tf.compat.v1.logging.info('writing to output path: %s', output_path)
    writers = [
        tf.io.TFRecordWriter(
            output_path + '-%05d-of-%05d.tfrecord' %  # noqa pylint: disable=C0209
            (i, num_shards)) for i in range(num_shards)
    ]

    images, img_to_obj_annotation, category_index = (
        _load_object_annotations(object_annotations_file))

    pool = multiprocessing.Pool()  # noqa pylint: disable=R1732
    total_num_annotations_skipped = 0
    log_total = {}
    cat_total = Counter()
    for idx, (_, tf_example, num_annotations_skipped, log_warnings, cats) in enumerate(
        pool.imap(_pool_create_tf_example, [(
            image,
            img_to_obj_annotation[image['id']],
            image_dir,
            category_index,
            include_masks) for image in images])):
        if idx % 100 == 0:
            tf.compat.v1.logging.info('On image %d of %d', idx, len(images))

        total_num_annotations_skipped += num_annotations_skipped
        log_total = _merge_log(log_total, log_warnings)
        cat_total.update(cats)
        writers[idx % num_shards].write(tf_example.SerializeToString())

    pool.close()
    pool.join()

    for writer in writers:
        writer.close()

    tf.compat.v1.logging.info(
        'Finished writing, skipped %d annotations.', total_num_annotations_skipped)
    return log_total, cat_total


@monitor_status(name="efficientdet", mode="data conversion")
def run_conversion(cfg):
    """Run data conversion."""
    log_dir = cfg.dataset_convert.output_dir

    # config output files
    tag = cfg.dataset_convert.tag or os.path.splitext(os.path.basename(cfg.dataset_convert.annotations_file))[0]
    output_path = os.path.join(cfg.dataset_convert.output_dir, tag)

    log_total, cat_total = _create_tf_record_from_coco_annotations(
        cfg.dataset_convert.annotations_file,
        cfg.dataset_convert.image_dir,
        output_path,
        cfg.dataset_convert.include_masks,
        num_shards=cfg.dataset_convert.num_shards)

    if log_total:
        with open(os.path.join(log_dir, f'{tag}_warnings.json'), "w", encoding='utf-8') as f:
            json.dump(log_total, f)

    status_logging.get_status_logger().categorical = {'num_objects': cat_total}


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="dataset_convert", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Convert COCO format json and images into TFRecords."""
    run_conversion(cfg)


if __name__ == '__main__':
    main()
