#!/bin/bash
# Description: Script responsible for reverting obfuscation.

set -e  # Exit on any error

# Setting the repo root for local build environment or CI.
[[ -z "${WORKSPACE}" ]] && REPO_ROOT='/workspace' || REPO_ROOT="${WORKSPACE}"

echo "Restoring the original project structure"

# Verify backup exists before proceeding
if [ ! -d "/orig_src" ]; then
    echo "ERROR: Backup directory /orig_src not found!"
    echo "Cannot revert obfuscation - original files are missing."
    exit 1
fi

# Check if nvidia_tao_tf2 directory exists
if [ ! -d "${REPO_ROOT}/nvidia_tao_tf2" ]; then
    echo "ERROR: Target directory ${REPO_ROOT}/nvidia_tao_tf2 not found!"
    exit 1
fi

# Remove obfuscated files
echo "Removing obfuscated files from ${REPO_ROOT}/nvidia_tao_tf2/"
rm -rf ${REPO_ROOT}/nvidia_tao_tf2/*

# Restore original files
echo "Restoring original files from /orig_src/"
if ! mv /orig_src/* ${REPO_ROOT}/nvidia_tao_tf2/; then
    echo "ERROR: Failed to restore original files!"
    exit 1
fi

echo "Successfully restored original source code"

# Clean up temporary directories
echo "Cleaning up temporary directories..."
rm -rf /orig_src
rm -rf /obf_src

# Clean up PyArmor runtime directories (more comprehensive)
echo "Cleaning up PyArmor runtime files..."
find ${REPO_ROOT} -name "pyarmor_runtime*" -type d -exec rm -rf {} + 2>/dev/null || true

# Note: Not removing /dist to preserve built wheels
echo "Revert completed successfully"
