# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Run scripts for functional tests."""

import argparse
import os
import subprocess
import sys

from utils import (
    CI, DOCKER_ROOT,
    ROOT_DIR
)

from utils import get_docker_command

TESTS = "pytest -color=yes -v" # -ss


def parse_command_line(args=sys.argv[1:]):
    """Parse command line args for running tests if needed."""
    parser = argparse.ArgumentParser(
        prog="run_tests_local",
        description="Simple script to run tests.",
        add_help=False)
    parser.add_argument(
        "--tag",
        default=None,
        help="Tag to the local base docker image.",
        type=str)
    parser.add_argument(
        "--mounts",
        default='/media/scratch.metropolis2:/home/scratch.metropolis2',
        help="Single mounting point.",
        type=str)
    args, unknown_args = parser.parse_known_args(args)
    return vars(args), unknown_args


def main(cl_args=None):
    """Simple function to run local tests."""
    args, unknown_args = parse_command_line(cl_args)
    manifest_file = os.path.join(ROOT_DIR, "docker/manifest.json")
    docker_command_prefix, docker_image = get_docker_command(
        manifest_file, args["tag"]
    )
    launcher_command = " ".join([TESTS] + unknown_args)
    print(launcher_command)
    if not CI:
        launcher_command = "{} -v {}:{} -v {} -e PYTHONPATH={} {} bash -c \'{}\'".format(
            docker_command_prefix, ROOT_DIR,
            DOCKER_ROOT, 
            args["mounts"],
            DOCKER_ROOT,
            docker_image, launcher_command)
        print(launcher_command)

    rc = subprocess.call(launcher_command, stdout=sys.stdout, shell=True)
    assert rc == 0 , "Functional tests failed."

if __name__=="__main__":
    main(sys.argv[1:])
