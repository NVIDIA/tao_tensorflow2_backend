# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.

"""TAO Toolkit Entrypoint Helper Modules."""

import importlib
import os
import pkgutil
import subprocess
import shlex
import sys
from time import time

from nvidia_tao_tf2.common.entrypoint import download_specs
from nvidia_tao_tf2.common.telemetry.nvml_utils import get_device_details
from nvidia_tao_tf2.common.telemetry.telemetry import send_telemetry_data


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
    assert len(gpu_ids) == num_gpus, (
        f"The number of GPUs ({gpu_ids}) must be the same as the number of GPU indices"
        f" ({num_gpus}) provided."
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
        "--experiment_spec",
        help="Path to the experiment spec file.",
        default=None
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        help="Number of GPUs to use. The default value is 1.",
        default=1
    )
    parser.add_argument(
        "-m",
        "--model_path",
        default=None,
        help="Path to a pre-trained model or model to continue training."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default=None,
        help="Path to where the output collaterals from this task is dropped."
    )
    parser.add_argument(
        '--gpu_index',
        type=int,
        nargs="+",
        help="The indices of the GPU's to be used.",
        default=None
    )
    parser.add_argument(
        "--num_processes",
        "-np",
        type=int,
        default=-1,
        help=("The number of horovod child processes to be spawned. "
              "Default is -1(equal to --gpus)."),
        required=False
    )
    parser.add_argument(
        "--mpirun-arg",
        type=str,
        default="-x NCCL_IB_HCA=mlx5_4,mlx5_6,mlx5_8,mlx5_10 -x NCCL_SOCKET_IFNAME=^lo,docker",
        help="Arguments for the mpirun command to run multi-node."
    )
    parser.add_argument(
        '--multi-node',
        action='store_true',
        default=False,
        help="Flag to enable to run multi-node training."
    )
    parser.add_argument(
        "--launch_cuda_blocking",
        action="store_true",
        default=False,
        help="Debug flag to add CUDA_LAUNCH_BLOCKING=1 to the command calls."
    )

    # Parse the arguments.
    return parser


def launch(parser, subtasks, multigpu_support=['train'], task="tao_tf2"):
    """Parse the command line and kick off the entrypoint.

    Args:

        parser (argparse.ArgumentParser): Parser object to define the command line args.
        subtasks (list): List of subtasks.
        multigpu_support (list): List of tasks that support --gpus > 1.
        task (str): Task entrypoint being called.
    """
    # Subtasks for a given model.
    parser = command_line_parser(parser, subtasks)

    cli_args = sys.argv[1:]
    args, unknown_args = parser.parse_known_args(cli_args)
    args = vars(args)

    scripts_args = ""
    if args["subtask"] not in ["download_specs"]:
        assert args["experiment_spec"], (
            f"Experiment spec file needs to be provided for this task: {args['subtask']}"
        )
        if not os.path.exists(args["experiment_spec"]):
            raise FileNotFoundError(f"Experiment spec file doesn't exist at {args['experiment_spec']}")
        path, name = os.path.split(args["experiment_spec"])
        if path != "":
            scripts_args += f" --config-path {path}"
        scripts_args += f" --config-name {name}"

    mpi_command = ""
    gpu_ids = args["gpu_index"]
    multi_node = args['multi_node']
    mpirun_arg = args['mpirun_arg']
    num_gpus = args["gpus"]
    if gpu_ids is None:
        gpu_ids = range(num_gpus)
    launch_cuda_blocking = args['launch_cuda_blocking']
    assert num_gpus > 0, "At least 1 GPU required to run any task."
    np = args["num_processes"]
    # np defaults to num_gpus if < 0
    if np < 0:
        np = num_gpus
    if num_gpus > 1:
        if not args["subtask"] in multigpu_support:
            raise NotImplementedError(
                f"This {args['subtask']} doesn't support multiGPU. Please set --gpus 1"
            )
        mpi_command = f'mpirun -np {np} --oversubscribe --bind-to none --allow-run-as-root -mca pml ob1 -mca btl ^openib'
        if multi_node:
            mpi_command += " " + mpirun_arg

    if args['subtask'] == "download_specs":
        if not args['output_dir']:
            raise RuntimeError(
                f"--output_dir is a mandatory arg for this subtask {args['subtask']}. "
                "Please set the output dir to a valid unix path."
            )
        scripts_args += f"target_data_dir={args['output_dir']}"
        scripts_args += f" source_data_dir={subtasks[args['subtask']]['source_data_dir']}"
        scripts_args += f" workflow={subtasks[args['subtask']]['workflow']}"

    script = subtasks[args['subtask']]["runner_path"]

    unknown_args_string = " ".join(unknown_args)
    task_command = f"python {script} {scripts_args} {unknown_args_string}"
    env_variables = ""
    if not multi_node:
        env_variables += set_gpu_info_single_node(num_gpus, gpu_ids)
    if launch_cuda_blocking:
        task_command = f"CUDA_LAUNCH_BLOCKING=1 {task_command}"
    run_command = f"{mpi_command} bash -c \'{env_variables} {task_command}\'"

    process_passed = True
    start = time()
    try:
        subprocess.run(
            shlex.split(run_command),
            shell=False,
            check=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
    except (KeyboardInterrupt, SystemExit):
        print("Command was interrupted.")
    except subprocess.CalledProcessError as e:
        process_passed = False
        if e.output is not None:
            print(f"TAO Toolkit task: {args['subtask']} failed with error:\n{e.output}")
    end = time()
    time_lapsed = end - start

    # Computing and sending telemetry data.
    try:
        gpu_data = []
        for device in get_device_details():
            gpu_data.append(device.get_config())
        print("Sending telemetry data.")
        send_telemetry_data(
            task,
            args["subtask"],
            gpu_data,
            num_gpus=num_gpus,
            time_lapsed=time_lapsed,
            pass_status=process_passed
        )
    except Exception as e:
        print("Telemetry data couldn't be sent, but the command ran successfully.")
        print(f"[Error]: {e}")

    if not process_passed:
        print("Execution status: FAIL")
        sys.exit(-1)  # returning non zero return code from the process.

    print("Execution status: PASS")
