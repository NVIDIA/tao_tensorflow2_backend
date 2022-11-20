# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

"""Define entrypoint to run tasks for classification."""

import argparse
from nvidia_tao_tf2.cv.classification import scripts

from nvidia_tao_tf2.common.entrypoint.entrypoint import get_subtasks, launch


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "classification",
        add_help=True,
        description="Train Adapt Optimize Toolkit entrypoint for classification"
    )

    # Build list of subtasks by inspecting the scripts package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(parser, subtasks, task="classification_tf2")


if __name__ == '__main__':
    main()
