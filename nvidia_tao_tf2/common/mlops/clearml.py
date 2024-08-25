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

"""Module to instantiate and return a clearml task."""

from datetime import datetime
import logging
import os

from clearml import Task

logger = logging.getLogger(__name__)


def check_clearml_init():
    """Check if ClearML has been properly initialized."""
    clearml_web_host = os.getenv("CLEARML_WEB_HOST", None)
    clearml_api_host = os.getenv("CLEARML_API_HOST", None)
    clearml_files_host = os.getenv("CLEARML_FILES_HOST", None)
    clearml_api_access_key = os.getenv("CLEARML_API_ACCESS_KEY", None)
    clearml_api_secret_key = os.getenv("CLEARML_API_SECRET_KEY", None)

    if not all([clearml_web_host, clearml_api_host, clearml_files_host, clearml_api_access_key, clearml_api_secret_key]):
        logger.warning("ClearML has not been initialized due to missing environment variables.")
        return False

    return True


def get_clearml_task(clearml_config, network_name: str, action: str = "train"):
    """Get clearml task.

    Args:
        clearml_config (protobuf): Configuration element for clearml task.
        network_name (str): Name of the network running the training.

    Returns
        task (clearml.Task): Task object.
    """
    time_string = datetime.now().strftime("%d/%y/%m_%H:%M:%S")
    task = None
    try:
        time_now = datetime.now().strftime("%d/%y/%m_%H:%M:%S")
        task_name = f"{clearml_config.task}_{time_string}" if clearml_config.task \
            else f"{network_name}_{action}_{time_now}"
        task = Task.init(
            project_name=clearml_config.project,
            task_name=task_name,
            deferred_init=clearml_config.deferred_init,
            reuse_last_task_id=clearml_config.reuse_last_task_id,
            continue_last_task=clearml_config.continue_last_task,
        )
        tao_base_container = os.getenv("TAO_DOCKER", None)
        if tao_base_container is not None:
            task.set_base_docker(tao_base_container)
        return task
    except Exception as e:
        logger.warning(
            "ClearML task init failed with error %s, but training will still continue.", e
        )
        return task
