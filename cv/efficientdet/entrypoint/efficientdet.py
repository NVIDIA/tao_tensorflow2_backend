# Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.

"""Define entrypoint to run tasks for efficientdet."""

import argparse
from cv.efficientdet import scripts

from common.entrypoint.entrypoint import get_subtasks, launch


def main():
    """Main entrypoint wrapper."""
    # Create parser for a given task.
    parser = argparse.ArgumentParser(
        "efficientdet_tf2",
        add_help=True,
        description="TAO Toolkit entrypoint for EfficientDet (TF2)"
    )

    # Build list of subtasks by inspecting the scripts package.
    subtasks = get_subtasks(scripts)

    # Parse the arguments and launch the subtask.
    launch(parser, subtasks)


if __name__ == '__main__':
    main()
