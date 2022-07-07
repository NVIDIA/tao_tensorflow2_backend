# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

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
            launch_command = '{} -v {}:{} {} {}'.format(
                docker_command_prefix, ROOT_DIR, DOCKER_ROOT, docker_image, command
            )
        sys.stdout.flush()
        rc = subprocess.call(launch_command, stdout=sys.stdout, shell=True)
        assert rc == 0


if __name__ == "__main__":
    main(sys.argv[1:])
