# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Export a classification model."""
import os
import logging

from common.hydra.hydra_runner import hydra_runner

from cv.classification.config.default_config import ExperimentConfig
from cv.classification.export.classification_exporter import Exporter
logger = logging.getLogger(__name__)


def run_export(cfg=None):
    """Export classification model to etlt."""
    exporter = Exporter(config=cfg)
    exporter.export()
    logger.info("Export finished successfully.")


spec_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra_runner(
    config_path=os.path.join(spec_root, "experiment_specs"),
    config_name="export", schema=ExperimentConfig
)
def main(cfg: ExperimentConfig) -> None:
    """Wrapper function for continuous training of classification application."""
    run_export(cfg=cfg)


if __name__ == "__main__":
    main()
