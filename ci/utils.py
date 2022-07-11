# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

"""Utilities to run tests for TAO."""

import json
import os


def configure_env():
    """Get the env configuration."""
    ci = False
    if os.getenv("CI_PROJECT_DIR", None) is not None:
        root_dir = os.getenv("CI_PROJECT_DIR", None)
        docker_root = root_dir
        ci = True
    else:
        # Directory for local test runs.
        root_dir = os.getenv(
            "NV_TAO_TF2_TOP",
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        docker_root = "/workspace/tao-tf2"
    return ci, docker_root, root_dir


CI, DOCKER_ROOT, ROOT_DIR = configure_env()
# Testing modules.
RCFILE = ".pylintrc"
TEST_MODULES = ['backbones', 'blocks', 'common', 'cv/efficientdet'] # 'cv', 'model_optimization']


def get_docker_information(manifest_file):
    """Loading the manifest to pick up the latest base docker."""
    assert os.path.exists(manifest_file), (
        "Manifest file doesn't exist at {}".format(
            manifest_file)
    )
    with open(manifest_file, "r") as m_file:
        docker_config = json.load(m_file)

    return docker_config["registry"], docker_config["repository"], docker_config["digest"]


def get_docker_command(manifest_file, tag):
    """Get formatted docker command."""
    docker_registry, docker_repository, docker_digest = get_docker_information(
        manifest_file)
    docker_image = "{}/{}@{}".format(
        docker_registry, docker_repository, docker_digest
    )
    if tag is not None:
        docker_image = "{}/{}:{}".format(
            docker_registry, docker_repository, tag
        )
    docker_command_prefix = "docker run --rm --gpus all"
    return docker_command_prefix, docker_image
