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

"""Instantiate the TAO-TF2 docker container for developers."""

import argparse
import os
import subprocess
import sys

from utils import CI, DOCKER_ROOT, ROOT_DIR, RCFILE, TEST_MODULES
from utils import get_docker_command

STATIC_TESTS = [
    # test cases.
    "pylint --rcfile {}".format(os.path.join(DOCKER_ROOT, RCFILE)),
    "pydocstyle --ignore=D400,D213,D203,D211,D4",
    "flake8 --ignore=E24,W504,E501",
]


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(prog="run_tests_local", description="Simple script to run tests.")
    parser.add_argument("--tag", default=None, help="Tag to the local base docker image.", type=str)
    return vars(parser.parse_args(args))


def main(cl_args=None):
    """Simple function to run local tests."""
    args = parse_command_line(cl_args)

    manifest_file = os.path.join(ROOT_DIR, "docker/manifest.json")
    docker_command_prefix, docker_image = get_docker_command(manifest_file, args["tag"])

    test_command = []
    for test in STATIC_TESTS:
        test_command.extend(["{} {}".format(test, " ".join(TEST_MODULES))])

    for command in test_command:
        launch_command = command
        print(launch_command)
        if not CI:
            print("Running test in local environment.")
            env_variables = "-e PYTHONPATH={}:$PYTHONPATH ".format("/workspace/tao-tf2")
            launch_command = '{} -v {}:{} {} {} {}'.format(
                docker_command_prefix, ROOT_DIR, DOCKER_ROOT, env_variables,
                docker_image, command
            )
        sys.stdout.flush()
        rc = subprocess.call(launch_command, stdout=sys.stdout, shell=True)
        assert rc == 0


if __name__ == "__main__":
    main(sys.argv[1:])
