#!/bin/bash
# Description: Script responsible for reverting obfuscation.

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/workspace/tao-tf2' || REPO_ROOT="${WORKSPACE}"

echo "Restoring the original project structure"
# Move the obf_src files.
rm -rf ${REPO_ROOT}/backbones
rm -rf ${REPO_ROOT}/blocks
rm -rf ${REPO_ROOT}/common
rm -rf ${REPO_ROOT}/cv
rm -rf ${REPO_ROOT}/model_optimization

# Move back the original files
mv /orig_src/* ${REPO_ROOT}/

# Remove the tmp folders.
rm -rf /dist
rm -rf /orig_src
rm -rf /obf_src
rm -rf ${REPO_ROOT}/pytransform_vax_001219
