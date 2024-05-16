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

"""Command line interface for generating changelogs."""

import argparse
from datetime import date, datetime, timedelta
import difflib
from io import StringIO
import os
import re
import subprocess
import sys
from tabulate import tabulate

# This will be generated every week.
INTERVAL = 7
DEFAULT_START_TIME = datetime.strftime(
    (date.today() - timedelta(days=INTERVAL)),
    "%m/%d/%y"
)
TAO_VERSION="3.22.05"
BUILD="1"
VALID_MODULES = {
    "TTS",
    "ConvAI",
    "NLP",
    "ASR",
    "AR",
    "PP",
    "CI",
    "TensorRT",
    "Converter",
    "Common",
    "Release",
    "PointPillars",
}


def parse_command_line(args=None):
    """Simple command line parser."""
    parser = argparse.ArgumentParser(
        prog="generate_changelog",
        description="Generate changelog from a given date."
    )
    parser.add_argument(
        "-d", "--date",
        help="Date in dd/mm/yyyy format",
        default=DEFAULT_START_TIME,
        type=str
    )
    parser.add_argument(
        "--build_type",
        help="Type of BUILD",
        default="Pre-Release",
        choices=["Pre-Release", "RC", "FC"]
    )
    parser.add_argument(
        "-o", "--output",
        help="Path to output file",
        required=False
    )
    return vars(parser.parse_args(args))


def header(message, level=1):
    """Format message as a header."""
    assert isinstance(message, str), (
        "The header must be a string."
    )
    return f'{"#"*level} {message}\n'


def generate_feature_list(module_name: str, feature_list: list):
    """Generate feature list from module log."""
    module_string = f"{header(module_name, level=3)}\n"
    for feature in feature_list:
        module_string += f"- {feature}\n"
    return module_string


def get_valid_module_name(module_name):
    "Get valid module name."
    return difflib.get_close_matches(module_name, VALID_MODULES)


def write_changelog_file(filename, module_logs, build_type="Pre-Release"):
    """Generate change log file."""
    read_mode = "w+"
    if os.path.exists(filename):
        read_mode = "a"
    file_content = ""
    with open(filename, read_mode) as mdfile:
        if read_mode == "w+":
            file_content = f"{header(message='CHANGELOG', level=1)}"

        version_string = header(
            message="TAO VERSION {version_id}-{build_id}".format(
                version_id=os.getenv("TAO_VERSION", TAO_VERSION),
                build_id=os.getenv("BUILD_NUMBER", BUILD)
            ),
            level=2
        )
        file_content += f"\n{version_string}\n"
        today = datetime.strftime(date.today(), "%m/%d/%y")
        table_data = {
            "BUILD TYPE": [build_type],
            "DATE": [today]
        }
        table = tabulate(table_data, headers="keys", tablefmt="github")
        file_content += f"{table}\n"   
        for module in module_logs.keys():
            file_content += f"\n{generate_feature_list(module, module_logs[module])}"
        mdfile.write(file_content)


def main(cl_args=None):
    """Generate changelog to a file."""
    # Get the repo root
    top_dir = os.environ.get("NV_TAO_TF2_TOP", os.path.abspath(os.getcwd()))
    parsed_args = parse_command_line(args=cl_args)
    from_date = parsed_args.get("date", DEFAULT_START_TIME)
    changelog_command = [
        "git",
        "log",
        "--oneline",
        "--pretty=\"%s\"",
        "--no-merges",
        f"--after={from_date}"
    ]
    git_log = subprocess.check_output(changelog_command)
    commit_string = StringIO(git_log.decode().strip())
    module = {}
    for row in commit_string.readlines():
        row = row.strip("\n")
        result = re.search(r"\[([A-Za-z0-9_]+)\]", row)
        if not result:
            continue
        key = result.group(1)
        matched = get_valid_module_name(key)
        valid_module = "Generic"
        if matched:
            valid_module = matched[0]
        if valid_module not in module.keys():
            module[valid_module] = []
        if valid_module != "Generic":
            body = row.replace(f"[{key}] ", "")
        else:
            body = row.replace("[", "\[")
            body = body.replace("]", "\]")
        # print(f"{key}, {body}")
        module[valid_module].append(body.strip("\""))

    # Write changelog md file.
    mdfile_path = os.path.join(
        os.getenv(
            "NV_AI_INFRA_TOP",
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ),
        "CHANGELOG.md"
    )

    if parsed_args["output"]:
        mdfile_path = os.path.join(parsed_args["output"], "TF2_CHANGELOG.md")

    write_changelog_file(
        mdfile_path, module,
        build_type=parsed_args["build_type"]
    )


if __name__=="__main__":
    main()

