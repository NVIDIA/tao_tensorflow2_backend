#!/bin/bash
# Description: Script responsible for generation of an obf_src wheel using pyarmor package.

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/workspace' || REPO_ROOT="${WORKSPACE}"
echo "Building from ${REPO_ROOT}"

echo "Installing required packages"
pip install pyarmor pyinstaller pybind11
echo "Registering pyarmor"
pyarmor -d reg ${REPO_ROOT}/release/docker/pyarmor-regfile-1219.zip || exit $?

echo "Clearing build and dists"
python ${REPO_ROOT}/setup.py clean --all
rm -rf dist/*
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo "Obfuscating the code using pyarmor"
pyarmor -d gen --recursive --output /obf_src/ ${REPO_ROOT}/nvidia_tao_tf2/ || exit $?

echo "Migrating codebase"
# Move sources to orig_src
rm -rf /orig_src
mkdir /orig_src
mv ${REPO_ROOT}/nvidia_tao_tf2/* /orig_src/

# PyArmor creates /obf_src/nvidia_tao_tf2/ structure, so we need to move from that subdirectory
echo "Moving obfuscated files from /obf_src/nvidia_tao_tf2/ to ${REPO_ROOT}/nvidia_tao_tf2/"
if [ -d "/obf_src/nvidia_tao_tf2" ]; then
    mv /obf_src/nvidia_tao_tf2/* ${REPO_ROOT}/nvidia_tao_tf2/
else
    echo "ERROR: Expected /obf_src/nvidia_tao_tf2/ directory not found!"
    echo "Contents of /obf_src/:"
    ls -la /obf_src/ || true
    exit 1
fi

# Move PyArmor runtime to project root for setup.py
echo "Moving PyArmor runtime to project root..."
if [ -d "/obf_src/pyarmor_runtime_001219" ]; then
    mv /obf_src/pyarmor_runtime_001219 ${REPO_ROOT}/
    echo "✓ Moved pyarmor_runtime_001219 to project root"
elif [ -d "${REPO_ROOT}/nvidia_tao_tf2/pyarmor_runtime_001219" ]; then
    mv ${REPO_ROOT}/nvidia_tao_tf2/pyarmor_runtime_001219 ${REPO_ROOT}/
    echo "✓ Moved pyarmor_runtime_001219 from nvidia_tao_tf2 to project root"
else
    echo "WARNING: pyarmor_runtime_001219 not found, checking for other runtime directories..."
    # PyArmor might create runtime with different names
    find /obf_src -name "pyarmor_runtime*" -type d -exec mv {} ${REPO_ROOT}/ \; 2>/dev/null
    find ${REPO_ROOT}/nvidia_tao_tf2 -name "pyarmor_runtime*" -type d -exec mv {} ${REPO_ROOT}/ \; 2>/dev/null
fi
