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
"""Inference with classification tensorrt engine."""

import os
import logging
import sys

import numpy as np
from PIL import ImageFile

from nvidia_tao_core.config.classification_tf2.default_config import ExperimentConfig

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner

from nvidia_tao_tf2.cv.classification.inferencer.trt_inferencer import TRTInferencer
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
    # Deprecated: DLFW 25.01 doesn't support tensorflow_quantization
    if sys.version_info >= (3, 12):
        logger.warning("DeprecationWarning: QAT is not supported after DLFW 25.01. Using normal training.")
        cfg.train.qat = False

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
