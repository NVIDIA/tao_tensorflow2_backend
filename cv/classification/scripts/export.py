# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""Export a classification model."""
import os
import logging

from common.decorators import monitor_status
from common.hydra.hydra_runner import hydra_runner

from cv.classification.config.default_config import ExperimentConfig
from cv.classification.export.classification_exporter import Exporter
logger = logging.getLogger(__name__)


@monitor_status(name='classification', mode='export')
def run_export(cfg=None):
    """Export classification model to etlt."""
    logger.setLevel(logging.DEBUG if cfg.verbose else logging.INFO)
    exporter = Exporter(config=cfg)
    exporter.export()


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
