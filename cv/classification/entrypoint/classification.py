# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

"""Define entrypoint to run tasks for classification."""

import argparse
from cv.classification import scripts

from common.entrypoint.entrypoint import get_subtasks, launch


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
    launch(parser, subtasks)


if __name__ == '__main__':
    main()
