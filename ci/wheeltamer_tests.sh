#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Description: Script testing the wheel using DL Framework's wheeltamer.
# Usage: sh wheeltamer_test.sh path_to_whl_file

WHEEL_PATH=$1
DIR="$(dirname "${WHEEL_PATH}")" ; WHEEL_FILENAME="$(basename "${WHEEL_PATH}")"

#Check if wheel file exists
if [ ! -f $WHEEL_PATH ]
then
    echo "ERROR: Couldn't find wheel in the dist/ folder"
    exit 0
fi
    echo "INFO: found EFF wheel: ${WHEEL_FILENAME}"


# License expected to be found in your package.
EXPECTED_PKG_LICENSE="NVIDIA Proprietary Software"

# Comma separated list of rules that are ignored (e.g "B301,B303,B304")
# See: https://bandit.readthedocs.io/en/latest/blacklists/blacklist_calls.html
# **MUST READ:** https://gitlab-master.nvidia.com/dl/pypi/Wheel-CI-CD#important-notice-do-not-skip-over-this-section 
# Example: SKIPPED_SECURITY_RULES="B301,B403,B404,B602,B603,B605,B607"
SKIPPED_SECURITY_RULES=""

# Pre-approved number of times where `# nosec` is used to skip security check.
# This method shall only be used on false alarms. Any pre-approved potentially
# risky code needs to go through the variable `SKIPPED_SECURITY_RULES`.
# **MUST READ:** https://gitlab-master.nvidia.com/dl/pypi/Wheel-CI-CD#important-notice-do-not-skip-over-this-section 
# Example: ALLOWED_NOSEC_COUNT="6"
ALLOWED_NOSEC_COUNT="0"

cd $DIR
# Run wheeltaimer in docker.
docker pull gitlab-master.nvidia.com:5005/dl/pypi/wheel-ci-cd:wheeltamer
docker run --rm --network=host \
    -e EXPECTED_PKG_LICENSE="${EXPECTED_PKG_LICENSE}" \
    -e SKIPPED_SECURITY_RULES="${SKIPPED_SECURITY_RULES}" \
    -e ALLOWED_NOSEC_COUNT="${ALLOWED_NOSEC_COUNT}" \
    -v $(pwd)/${WHEEL_FILENAME}:/workspace/${WHEEL_FILENAME} \
    gitlab-master.nvidia.com:5005/dl/pypi/wheel-ci-cd:wheeltamer

