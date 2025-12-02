#!/bin/bash

set -ex

if [ -f "$(dirname "${BASH_SOURCE[0]}")/detect_rocm_path.sh" ]; then
  source "$(dirname "${BASH_SOURCE[0]}")/detect_rocm_path.sh"
else
  ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
fi

echo "Installing amdsmi from: ${ROCM_PATH}/share/amd_smi"
cd ${ROCM_PATH}/share/amd_smi && pip install .
