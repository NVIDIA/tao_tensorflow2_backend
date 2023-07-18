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

"""Define entrypoint to run tasks for efficientdet."""

import argparse
from nvidia_tao_tf2.cv.efficientdet import scripts

from nvidia_tao_tf2.common.entrypoint.entrypoint import get_subtasks, launch


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
    launch(
        parser, subtasks,
        multigpu_support=['train', 'evaluate'],
        task="efficientdet_tf2"
    )


if __name__ == '__main__':
    main()
