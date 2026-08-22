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

KAGGLE_CREDENTIALS_FILE="${HOME}/.kaggle/kaggle.json"
if [ -f "${KAGGLE_CREDENTIALS_FILE}" ]; then
    file_kaggle_username="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["username"])' "${KAGGLE_CREDENTIALS_FILE}")"
    file_kaggle_key="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["key"])' "${KAGGLE_CREDENTIALS_FILE}")"
    KAGGLE_USERNAME="${KAGGLE_USERNAME:-${file_kaggle_username}}"
    KAGGLE_KEY="${KAGGLE_KEY:-${file_kaggle_key}}"
fi
if [ -n "${KAGGLE_KEY:-}" ]; then
    KAGGLE_API_TOKEN="${KAGGLE_API_TOKEN:-${KAGGLE_KEY}}"
fi
export KAGGLE_USERNAME KAGGLE_KEY KAGGLE_API_TOKEN

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
if [ -n "${KAGGLE_USERNAME:-}" ]; then
    echo "Notebook: https://www.kaggle.com/code/${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}"
else
    echo "Notebook slug: ${KAGGLE_KERNEL_SLUG}"
fi
if [ -n "${KAGGLE_DATASET_REF}" ]; then
    echo "Source dataset: https://www.kaggle.com/datasets/${KAGGLE_DATASET_REF}"
fi

if command -v kaggle >/dev/null 2>&1 && [ -n "${KAGGLE_USERNAME:-}" ]; then
    kaggle kernels status "${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}" || true
fi
