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

"""TAO Toolkit Entrypoint Helper Modules."""

import re
import ast
import importlib
import os
import pkgutil
import subprocess
import shlex
import sys
from time import time
from contextlib import contextmanager

import yaml

from nvidia_tao_tf2.common.entrypoint import download_specs
from nvidia_tao_tf2.common.telemetry.nvml_utils import get_device_details
from nvidia_tao_core.telemetry.telemetry import send_telemetry_data
import logging as _logging

_logging.basicConfig(
    format='[%(asctime)s - TAO Toolkit - %(name)s - %(levelname)s] %(message)s',
    level='INFO'
)
logging = _logging


def get_subtasks(package):
    """Get supported subtasks for a given task.

    This function lists out the python tasks in a folder.

    Returns:
        subtasks (dict): Dictionary of files.
    """
    module_path = package.__path__
    modules = {}

    # Collect modules dynamically.
    for _, task, is_package in pkgutil.walk_packages(module_path):
        if is_package:
            continue
        module_name = package.__name__ + '.' + task
        module_details = {
            "module_name": module_name,
            "runner_path": os.path.abspath(importlib.import_module(module_name).__file__),
        }
        modules[task] = module_details

    # Add new command for copying specs.
    modules["download_specs"] = {
        "source_data_dir": os.path.join(os.path.dirname(module_path[0]), "experiment_specs"),
        "runner_path": os.path.abspath(importlib.import_module(download_specs.__name__).__file__),
        "workflow": package.__name__.split(".")[0]
    }
    return modules


def check_valid_gpus(num_gpus, gpu_ids):
    """Check if the number of GPU's called and IDs are valid.

    This function scans the machine using the nvidia-smi routine to find the
    number of GPU's and matches the id's and num_gpu's accordingly.

    Once validated, it finally also sets the CUDA_VISIBLE_DEVICES env variable.

    Args:
        num_gpus (int): Number of GPUs alloted by the user for the job.
        gpu_ids (list(int)): List of GPU indices used by the user.

    Returns:
        No explicit returns
    """
    # Ensure the gpu_ids are all different, and sorted
    gpu_ids = sorted(list(set(gpu_ids)))
    assert num_gpus > 0, "At least 1 GPU required to run any task."
    num_gpus_available = str(subprocess.check_output(["nvidia-smi", "-L"])).count("UUID")
    max_id = max(gpu_ids)
    assert min(gpu_ids) >= 0, (
        "GPU ids cannot be negative."
    )
    assert max_id < num_gpus_available and num_gpus <= num_gpus_available, (
        "Checking for valid GPU ids and num_gpus."
    )
    cuda_visible_devices = ",".join([str(idx) for idx in gpu_ids])
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices


def set_gpu_info_single_node(num_gpus, gpu_ids):
    """Set gpu environment variable for single node."""
    check_valid_gpus(num_gpus, gpu_ids)

    env_variable = ""
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    if visible_devices is not None:
        env_variable = f" CUDA_VISIBLE_DEVICES={visible_devices}"
    return env_variable


def command_line_parser(parser, subtasks):
    """Build command line parser."""
    parser.add_argument(
        'subtask',
        default='train',
        choices=subtasks.keys(),
        help="Subtask for a given task/model.",
    )
    parser.add_argument(
        "-e",
        "--experiment_spec_file",
        help="Path to the experiment spec file.",
        default=None
    )

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


@contextmanager
def dual_output(log_file=None):
    """Context manager to handle dual output redirection for subprocess.

    Args:
    - log_file (str, optional): Path to the log file. If provided, output will be
      redirected to both sys.stdout and the specified log file. If not provided,
      output will only go to sys.stdout.

    Yields:
    - stdout_target (file object): Target for stdout output (sys.stdout or log file).
    - log_target (file object or None): Target for log file output, or None if log_file
      is not provided.
    """
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            yield sys.stdout, f
    else:
        yield sys.stdout, None


def launch(args, unknown_args, subtasks, multigpu_support=['train'], network="tao_tf2"):
    """Parse the command line and kick off the entrypoint.

    Args:
        args: Known command line arguments dictionary.
        unknown_args: Unknown command line arguments string.
        subtasks (list): List of subtasks.
        multigpu_support (list): List of tasks that support --gpus > 1.
        network (str): network entrypoint being called.
    """
    # Subtasks for a given model.
    scripts_args = ""
    if args['subtask'] not in ["download_specs"]:
        assert args['experiment_spec_file'], (
            f"Experiment spec file needs to be provided for this network:{args['subtask']}"
        )
        if not os.path.exists(args['experiment_spec_file']):
            raise FileNotFoundError(f"Experiment spec file doesn't exist at {args['experiment_spec_file']}")
        path, name = os.path.split(args['experiment_spec_file'])
        if path != "":
            scripts_args += f" --config-path {path}"
        scripts_args += f" --config-name {name}"

    # This enables a results_dir arg to be passed from the microservice side,
    # but there is no --results_dir cmdline arg. Instead, the spec field must be used
    if "results_dir" in args:
        scripts_args += " results_dir=" + args["results_dir"]

    # Pass unknown args to call
    unknown_args_as_str = " " + " ".join(unknown_args)

    # Precedence these settings: cmdline > specfile > default
    overrides = ["num_gpus", "gpu_ids", "cuda_blocking", "mpi_args", "multi_node", "num_processes"]
    num_gpus = 1
    gpu_ids = [0]
    launch_cuda_blocking = False
    mpi_args = "-x NCCL_IB_HCA=mlx5_4,mlx5_6,mlx5_8,mlx5_10 -x NCCL_SOCKET_IFNAME=^lo,docker"
    multi_node = False
    np = -1

    # Parsing cmdline override
    if any(arg in unknown_args_as_str for arg in overrides):
        if "num_gpus" in unknown_args_as_str:
            num_gpus = int(unknown_args_as_str.split('num_gpus=')[1].split()[0])
        if "gpu_ids" in unknown_args_as_str:
            gpu_ids = ast.literal_eval(unknown_args_as_str.split('gpu_ids=')[1].split()[0])
        if "cuda_blocking" in unknown_args_as_str:
            launch_cuda_blocking = ast.literal_eval(unknown_args_as_str.split('cuda_blocking=')[1].split()[0])
        if "mpi_args" in unknown_args_as_str:
            mpi_args = ""
            mpi_arg_list = unknown_args_as_str.split('mpi_args.')[1:]
            for mpi_arg in mpi_arg_list:
                var, val = mpi_arg.strip().split("=")
                mpi_args += f"-x {var.upper()}={val} "
        if "multi_node" in unknown_args_as_str:
            multi_node = ast.literal_eval(unknown_args_as_str.split('multi_node=')[1].split()[0])
        if "num_processes" in unknown_args_as_str:
            np = int(unknown_args_as_str.split('num_processes=')[1].split()[0])
    # If no cmdline override, look at specfile
    else:
        with open(args["experiment_spec_file"], 'r') as spec:  # pylint: disable=W1514
            exp_config = yaml.safe_load(spec)
            if 'num_gpus' in exp_config:
                num_gpus = exp_config['num_gpus']
            if 'gpu_ids' in exp_config:
                gpu_ids = exp_config['gpu_ids']
            if "cuda_blocking" in exp_config:
                launch_cuda_blocking = exp_config['cuda_blocking']
            if "mpi_args" in exp_config:
                mpi_args = ""
                mpi_arg_dict = exp_config['mpi_args']
                for var, val in mpi_arg_dict.items():
                    mpi_args += f"-x {var.upper()}={val} "
            if "multi_node" in unknown_args_as_str:
                multi_node = exp_config['multi_node']

    if num_gpus != len(gpu_ids):
        logging.info(f"The number of GPUs ({num_gpus}) must be the same as the number of GPU indices ({gpu_ids}) provided.")
        num_gpus = max(num_gpus, len(gpu_ids))
        gpu_ids = list(range(num_gpus)) if len(gpu_ids) != num_gpus else gpu_ids
        logging.info(f"Using GPUs {gpu_ids} (total {num_gpus})")

    mpi_command = ""
    # np defaults to num_gpus if < 0
    if np < 0:
        np = num_gpus
    if num_gpus > 1:
        if not args['subtask'] in multigpu_support:
            raise NotImplementedError(
                f"This {args['subtask']} doesn't support multiGPU."
            )
        mpi_command = f'mpirun -np {np} --oversubscribe --bind-to none --allow-run-as-root -mca pml ob1 -mca btl ^openib'
        if multi_node:
            mpi_command += " " + mpi_args

    script = subtasks[args['subtask']]["runner_path"]

    log_file = ""
    if os.getenv('JOB_ID'):
        logs_dir = os.getenv('TAO_MICROSERVICES_TTY_LOG', '/results')
        log_file = f"{logs_dir}/{os.getenv('JOB_ID')}/microservices_log.txt"

    task_command = f"python {script} {scripts_args} {unknown_args_as_str}"
    env_variables = "TF_USE_LEGACY_KERAS=1"
    if not multi_node:
        env_variables += set_gpu_info_single_node(num_gpus, gpu_ids)
    if launch_cuda_blocking:
        task_command = f"CUDA_LAUNCH_BLOCKING=1 {task_command}"
    run_command = f"{mpi_command} bash -c \'{env_variables} {task_command}\'"

    process_passed = False
    start = time()
    progress_bar_pattern = re.compile(r"^(?!.*(Average Precision|Average Recall)).*\[.*\].*")

    try:
        # Run the script.
        with dual_output(log_file) as (stdout_target, log_target):
            with subprocess.Popen(
                shlex.split(run_command),
                env=os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,  # Line-buffered
                universal_newlines=True  # Text mode
            ) as proc:
                last_progress_bar_line = None

                for line in proc.stdout:
                    # Check if the line contains \r or matches the progress bar pattern
                    if '\r' in line or progress_bar_pattern.search(line):
                        last_progress_bar_line = line.strip()
                        # Print the progress bar line to the terminal
                        stdout_target.write('\r' + last_progress_bar_line)
                        stdout_target.flush()
                    else:
                        # Write the final progress bar line to the log file before a new log line
                        if last_progress_bar_line:
                            if log_target:
                                log_target.write(last_progress_bar_line + '\n')
                                log_target.flush()
                            last_progress_bar_line = None
                        stdout_target.write(line)
                        stdout_target.flush()
                        if log_target:
                            log_target.write(line)
                            log_target.flush()

                proc.wait()  # Wait for the process to complete
                # Write the final progress bar line after process completion
                if last_progress_bar_line and log_target:
                    log_target.write(last_progress_bar_line + '\n')
                    log_target.flush()
                if proc.returncode == 0:
                    process_passed = True
    except (KeyboardInterrupt, SystemExit) as e:
        logging.info("Command was interrupted due to ", e)
        process_passed = True
    except subprocess.CalledProcessError as e:
        process_passed = False
        if e.output is not None:
            logging.info(e.output)

    end = time()
    time_lapsed = end - start

    # Computing and sending telemetry data.
    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        logging.info("Sending telemetry data.")
        send_telemetry_data(
            network,
            args['subtask'],
            gpu_data,
            num_gpus=num_gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        logging.warning("Telemetry data couldn't be sent, but the command ran successfully.")
        logging.warning(f"[Error]: {e}")

    if not process_passed:
        logging.warning("Execution status: FAIL")
        return False

    logging.info("Execution status: PASS")
    return True
