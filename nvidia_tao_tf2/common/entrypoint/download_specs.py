# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

"""Common entrypoint to download default specs."""

import logging
import os
import shutil

from omegaconf import MISSING
from dataclasses import dataclass

from nvidia_tao_tf2.common.hydra.hydra_runner import hydra_runner

logger = logging.getLogger()


@dataclass
class DefaultConfig:
    """This is a structured config for downloading default spec files."""

    # Input folder where the default configs are.
    source_data_dir: str = MISSING

    # Output folder path
    target_data_dir: str = MISSING

    # Name of the worflow.
    workflow: str = MISSING


spec_path = os.path.dirname(os.path.abspath(__file__))


@hydra_runner(config_path=spec_path, config_name="download_specs", schema=DefaultConfig)
def main(cfg: DefaultConfig) -> None:
    """Script to download default specs."""
    if os.path.exists(cfg.target_data_dir):
        if os.listdir(cfg.target_data_dir):
            raise FileExistsError(
                f"The target directory, `{cfg.target_data_dir}` already has files in it."
                "Please empty this directory in order to avoid overwriting the default specs."
            )
    else:
        os.makedirs(cfg.target_data_dir)

    names = [item for item in os.listdir(cfg.source_data_dir) if item.endswith(".yaml")]

    for spec in names:
        srcname = os.path.join(cfg.source_data_dir, spec)
        dstname = os.path.join(cfg.target_data_dir, spec)
        shutil.copy2(srcname, dstname)

    logger.info(
        "Default specification files for {} downloaded to '{}'".format(cfg.workflow, cfg.target_data_dir)  # noqa: pylint: disable=C0209
    )


if __name__ == "__main__":
    main()
