# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""EfficientDet standalone inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import numpy as np
from PIL import Image

from numba import cuda

import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.util import deprecation

from cv.efficientdet.config.hydra_runner import hydra_runner
from cv.efficientdet.config.default_config import ExperimentConfig
from cv.efficientdet.exporter.image_batcher import ImageBatcher
from cv.efficientdet.inferencer import inference, inference_trt
from cv.efficientdet.model.efficientdet import efficientdet
from cv.efficientdet.utils import keras_utils
from cv.efficientdet.utils import hparams_config
from cv.efficientdet.utils.config_utils import generate_params_from_cfg
from cv.efficientdet.visualize import vis_utils

deprecation._PRINT_DEPRECATION_WARNINGS = False

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'
supported_img_format = ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']


def get_label_dict(label_txt):
    """Create label dict from txt file."""

    with open(label_txt, 'r') as f:
        labels = f.readlines()
        return {i+1 : label[:-1] for i, label in enumerate(labels)}


def batch_generator(iterable, batch_size=1):
    """Load a list of image paths in batches.

    Args:
        iterable: a list of image paths
        n: batch size
    """
    total_len = len(iterable)
    for ndx in range(0, total_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, total_len)]


def infer_tlt(cfg, label_id_mapping, min_score_thresh):
    """Launch EfficientDet TLT model Inference."""
    # disable_eager_execution()
    tf.autograph.set_verbosity(0)
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

    # Parse and update hparams
    config = hparams_config.get_detection_config(cfg['model_config']['model_name'])
    config.update(generate_params_from_cfg(config, cfg, mode='infer'))
    params = config.as_dict()
    image_dir = cfg['inference_config']['image_dir']

    # build model
    # TODO(@yuw): verify channels_first training or force last
    input_shape = list(config.image_size) + [3] \
        if config.data_format == 'channels_last' else [3] + list(config.image_size)
    _, model = efficientdet(input_shape, training=False, config=config)
    model.summary()
    keras_utils.restore_ckpt(
        model,
        cfg.inference_config.model_path,
        config.moving_average_decay,
        steps_per_epoch=0, skip_mismatch=False, expect_partial=True)

    infer_model = inference.InferenceModel(model, config.image_size, params,
                                           cfg.inference_config.batch_size,
                                           label_id_mapping=label_id_mapping,
                                           min_score_thresh=min_score_thresh,
                                           max_boxes_to_draw=100) # TODO(@yuw): make it configurable
    imgpath_list = [os.path.join(image_dir, imgname)
                    for imgname in sorted(os.listdir(image_dir))
                    if os.path.splitext(imgname)[1].lower()
                    in supported_img_format]
    for image_paths in batch_generator(imgpath_list, cfg.inference_config.batch_size):
        infer_model.visualize_detections(
            image_paths,
            cfg.inference_config.output_dir,
            cfg.inference_config.dump_label)
    print("Inference finished.")


def infer_trt(cfg, label_id_mapping, min_score_thresh):
    """Run trt inference."""
    output_dir = os.path.realpath(cfg.inference_config.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    trt_infer = inference_trt.TensorRTInfer(
        cfg.inference_config.model_path,
        label_id_mapping,
        min_score_thresh)
    trt_infer.visualize_detections(
        cfg.inference_config.image_dir,
        cfg.inference_config.output_dir,
        cfg.inference_config.dump_label)
    print("Finished Processing")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="inference", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig):
    """Wrapper function for EfficientDet inference.
    """
    label_id_mapping = {}
    if cfg.eval_config.label_map:
        label_id_mapping = get_label_dict(cfg.eval_config.label_map)

    if cfg.inference_config.model_path.endswith('.engine'):
        print("Running inference with TensorRT engine...")
        infer_trt(cfg, label_id_mapping, cfg.eval_config.min_score_thresh or 0.4)
    else: # TODO(@yuw): eff cfg.inference_config.model_path.endswith('.tlt'):
        print("Running inference with saved_model...")
        infer_tlt(cfg, label_id_mapping, cfg.eval_config.min_score_thresh or 0.4)


if __name__ == '__main__':
    main()