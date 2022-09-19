# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Inference and metrics computation code using a loaded model."""

import logging
import os

import numpy as np
import pandas as pd

from common.hydra.hydra_runner import hydra_runner

from cv.classification.inferencer.keras_inferencer import KerasInferencer
from cv.classification.config.default_config import ExperimentConfig

logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_FORMAT = ['.jpg', '.png', '.jpeg']


def run_inference(cfg):
    """Inference on an image/directory using a pretrained model file.

    Args:
        args: Dictionary arguments containing parameters defined by command
              line parameters.
    Log:
        Image Mode:
            print classifier output
        Directory Mode:
            write out a .csv file to store all the predictions
    """
    predictions = []
    image_depth = cfg['model']['input_image_depth']
    assert image_depth in [8, 16], "Only 8-bit and 16-bit images are supported"
    inferencer = KerasInferencer(cfg['infer']['model_path'],
                                key=cfg.key,
                                img_mean=list(cfg['train']['image_mean']),
                                img_depth=image_depth)

    for img_name in sorted(os.listdir(cfg['infer']['image_dir'])):
        _, ext = os.path.splitext(img_name)
        if ext.lower() in SUPPORTED_IMAGE_FORMAT:
            result = inferencer.infer_single(
                os.path.join(cfg['infer']['image_dir'], img_name))
            predictions.append(np.argmax(result))
            break
    print(os.path.join(cfg['infer']['image_dir'], img_name),predictions)



spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="infer", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application.
    """
    run_inference(cfg)
    logger.info("Inference finished successfully.")


if __name__ == "__main__":
    main()
