#!/bin/bash
#
# Deploy "The Duck" ARC3 harness to Kaggle via `make kaggle-duck`.
#
# Run this inside the duck-kaggle-dev container; the repository is mounted
# at /workspace/duck-harness. See AGENTS.md -> "Running `make kaggle-duck`".
#
# Defaults (matching the Makefile target):
#   RUN_NAME=duck-harness-kaggle
#   KAGGLE_KERNEL_SLUG=taaf-<RUN_NAME>
#   KAGGLE_DATASET_REF=<KAGGLE_USERNAME>/taaf-kaggle-source-<RUN_NAME>
#   KAGGLE_ACCELERATOR=NvidiaRtxPro6000
#   KAGGLE_DRY_RUN=false
#   DEPLOYMENT_WAIT=false
#
# Override any value by exporting it first, e.g.:
#   RUN_NAME=duck-harness-20260816 ./kaggle-duck.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_NAME="${RUN_NAME:-duck-harness-kaggle}"
KAGGLE_KERNEL_SLUG="${KAGGLE_KERNEL_SLUG:-taaf-${RUN_NAME}}"
KAGGLE_DATASET_REF="${KAGGLE_DATASET_REF:-}"
KAGGLE_ACCELERATOR="${KAGGLE_ACCELERATOR:-NvidiaRtxPro6000}"
KAGGLE_DRY_RUN="${KAGGLE_DRY_RUN:-false}"
DEPLOYMENT_WAIT="${DEPLOYMENT_WAIT:-false}"

if [ -z "${KAGGLE_DATASET_REF}" ] && [ -n "${KAGGLE_USERNAME:-}" ]; then
    KAGGLE_DATASET_REF="${KAGGLE_USERNAME}/taaf-kaggle-source-${RUN_NAME}"
fi

make_args=(RUN_NAME="${RUN_NAME}" KAGGLE_KERNEL_SLUG="${KAGGLE_KERNEL_SLUG}")
if [ -n "${KAGGLE_DATASET_REF}" ]; then
    make_args+=(KAGGLE_DATASET_REF="${KAGGLE_DATASET_REF}")
fi
make_args+=(
    KAGGLE_ACCELERATOR="${KAGGLE_ACCELERATOR}"
    KAGGLE_DRY_RUN="${KAGGLE_DRY_RUN}"
    DEPLOYMENT_WAIT="${DEPLOYMENT_WAIT}"
)

make kaggle-duck "${make_args[@]}"

if [ "${KAGGLE_DRY_RUN}" = "true" ]; then
    exit 0
fi

echo
echo "Notebook: https://www.kaggle.com/code/${KAGGLE_KERNEL_SLUG}"
if [ -n "${KAGGLE_DATASET_REF}" ]; then
    echo "Source dataset: https://www.kaggle.com/datasets/${KAGGLE_DATASET_REF}"
fi

if [ -f "${HOME}/.kaggle/kaggle.json" ]; then
    eval "$(python3 -c "import json; d=json.load(open('${HOME}/.kaggle/kaggle.json')); print('export KAGGLE_USERNAME=%r' % d['username']); print('export KAGGLE_KEY=%r' % d['key']); print('export KAGGLE_API_TOKEN=%r' % d['key'])")"
fi
if command -v kaggle >/dev/null 2>&1; then
    kaggle kernels status "${KAGGLE_KERNEL_SLUG}" || true
fi
