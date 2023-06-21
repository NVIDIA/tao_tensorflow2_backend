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

"""Inference and metrics computation code using a loaded model."""

import json
import logging
import os
import tqdm

import numpy as np
import pandas as pd

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner
from nvidia_tao_tf2.common.decorators import monitor_status
from nvidia_tao_tf2.common.utils import update_results_dir

from nvidia_tao_tf2.cv.classification.inferencer.keras_inferencer import KerasInferencer
from nvidia_tao_tf2.cv.classification.config.default_config import ExperimentConfig
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', level='INFO')
logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_FORMAT = ['.jpg', '.png', '.jpeg']


@monitor_status(name='classification', mode='inference')
def run_inference(cfg):
    """Inference on a directory of images using a pretrained model file.

    Args:
        cfg: Hydra config.
    Log:
        Directory Mode:
            write out a .csv file to store all the predictions
    """
    logger.setLevel(logging.INFO)
    if not os.path.exists(cfg.results_dir):
        os.makedirs(cfg.results_dir, exist_ok=True)
    result_csv_path = os.path.join(cfg.results_dir, 'result.csv')
    with open(cfg.inference.classmap, "r", encoding='utf-8') as cm:
        class_dict = json.load(cm)
    reverse_mapping = {v: k for k, v in class_dict.items()}

    image_depth = cfg.model.input_image_depth
    assert image_depth in [8, 16], "Only 8-bit and 16-bit images are supported"
    interpolation = cfg.model.resize_interpolation_method
    if cfg.dataset.augmentation.enable_center_crop:
        interpolation += ":center"
    inferencer = KerasInferencer(
        cfg.inference.checkpoint,
        key=cfg.encryption_key,
        img_mean=list(cfg.dataset.image_mean),
        preprocess_mode=cfg.dataset.preprocess_mode,
        interpolation=interpolation,
        img_depth=image_depth,
        data_format=cfg.data_format)
    predictions = []

    imgpath_list = [os.path.join(root, filename)
                    for root, subdirs, files in os.walk(cfg.inference.image_dir)
                    for filename in files
                    if os.path.splitext(filename)[1].lower()
                    in SUPPORTED_IMAGE_FORMAT
                    ]

    for img_name in tqdm.tqdm(imgpath_list):
        raw_predictions = inferencer.infer_single(img_name)
        class_index = np.argmax(raw_predictions)
        class_labels = reverse_mapping[class_index]
        class_conf = np.max(raw_predictions)
        predictions.append((img_name, class_labels, class_conf))

    with open(result_csv_path, 'w', encoding='utf-8') as csv_f:
        # Write predictions to file
        df = pd.DataFrame(predictions)
        df.to_csv(csv_f, header=False, index=False)
    logger.info("The inference result is saved at: %s", result_csv_path)


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="infer", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for classification inference."""
    cfg = update_results_dir(cfg, 'inference')
    run_inference(cfg)


if __name__ == "__main__":
    main()
