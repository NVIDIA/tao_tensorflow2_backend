# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Inference and metrics computation code using a loaded model."""

import json
import logging
import os
import tqdm

import numpy as np
import pandas as pd

from common.decorators import monitor_status
from common.hydra.hydra_runner import hydra_runner

from cv.classification.inferencer.keras_inferencer import KerasInferencer
from cv.classification.config.default_config import ExperimentConfig

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
    logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
    result_csv_path = os.path.join(cfg.results_dir, 'result.csv')
    assert os.path.exists(cfg.results_dir), "The results_dir doesn't exist."
    assert os.path.exists(cfg.inference.classmap), "inference.classmap doesn't exist."
    assert os.path.exists(cfg.inference.model_path), "inference.model_path doesn't exist."
    with open(cfg.inference.classmap, "r", encoding='utf-8') as cm:
        class_dict = json.load(cm)
    reverse_mapping = {v: k for k, v in class_dict.items()}

    image_depth = cfg.model.input_image_depth
    assert image_depth in [8, 16], "Only 8-bit and 16-bit images are supported"
    interpolation = cfg.model.resize_interpolation_method
    if cfg.augment.enable_center_crop:
        interpolation += ":center"
    inferencer = KerasInferencer(
        cfg.inference.model_path,
        key=cfg.key,
        img_mean=list(cfg.data.image_mean),
        preprocess_mode=cfg.data.preprocess_mode,
        interpolation=interpolation,
        img_depth=image_depth)
    predictions = []
    for img_name in tqdm.tqdm(sorted(os.listdir(cfg.inference.image_dir))):
        _, ext = os.path.splitext(img_name)
        if ext.lower() in SUPPORTED_IMAGE_FORMAT:
            raw_predictions = inferencer.infer_single(
                os.path.join(cfg.inference.image_dir, img_name))
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
    """Wrapper function for continuous training of classification application."""
    run_inference(cfg)


if __name__ == "__main__":
    main()
