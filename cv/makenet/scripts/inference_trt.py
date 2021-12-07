# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
"""Inference with classification tensorrt engine."""

import os
from functools import partial
import logging

import numpy as np
from PIL import ImageFile

from makenet.inferencer.trt_inferencer import TRTInferencer
from makenet.config.hydra_runner import hydra_runner
from makenet.config.default_config import ExperimentConfig
ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_FORMAT = ['.jpg', '.png', '.jpeg']


def run_inference(cfg):
    """Wrapper function to run evaluation of MakeNet model.

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
    inferencer = TRTInferencer(cfg['infer_config']['model_path'])

    for img_name in os.listdir(cfg['infer_config']['image_dir']):
        _, ext = os.path.splitext(img_name)
        if ext.lower() in SUPPORTED_IMAGE_FORMAT:
            result = inferencer.infer_single(
                os.path.join(cfg['infer_config']['image_dir'], img_name))
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
    """Wrapper function for continuous training of MakeNet application.
    """
    run_inference(cfg)
    logger.info("Inference finished successfully.")


if __name__ == '__main__':
    main()
