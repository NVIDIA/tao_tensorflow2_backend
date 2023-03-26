# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
"""Inference with classification tensorrt engine."""

import os
import logging

import numpy as np
from PIL import ImageFile

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner

from nvidia_tao_tf2.cv.classification.inferencer.trt_inferencer import TRTInferencer
from nvidia_tao_tf2.cv.classification.config.default_config import ExperimentConfig
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_FORMAT = ['.jpg', '.png', '.jpeg']


def run_inference(cfg):
    """Wrapper function to run evaluation of classification model.

    Args:
       Dictionary arguments containing parameters parsed in the main function.
    """
    # Set up logger verbosity.
    verbosity = 'INFO'
    # Configure the logger.
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        level=verbosity)
    # set backend
    # initialize()
    predictions = []
    inferencer = TRTInferencer(cfg['inference']['checkpoint'], batch_size=1,
                               data_format=cfg['data_format'],
                               img_depth=cfg['model']['input_image_depth'])

    for img_name in os.listdir(cfg['inference']['image_dir']):
        _, ext = os.path.splitext(img_name)
        if ext.lower() in SUPPORTED_IMAGE_FORMAT:
            result = inferencer.infer_single(
                os.path.join(cfg['inference']['image_dir'], img_name))
            # print(result)
            predictions.append(np.argmax(result))
            break
    print(predictions)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="infer", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application."""
    run_inference(cfg)
    logger.info("Inference finished successfully.")


if __name__ == '__main__':
    main()
