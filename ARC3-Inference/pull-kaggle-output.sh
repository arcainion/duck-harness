#!/bin/bash
# Pull the latest published Duck Kaggle notebook output for local analysis.

set -euo pipefail

KAGGLE_KERNEL_REF="${KAGGLE_KERNEL_REF:-arcainionprime/taaf-duck-harness-kaggle}"
KAGGLE_OUTPUT_DIR="${KAGGLE_OUTPUT_DIR:-/workspace/duck-harness/results}"
CHECK_STATUS=true

usage() {
    cat <<'EOF'
Usage: bash pull-kaggle-output.sh [options]

Options:
  -k, --kernel REF   Kaggle kernel reference (owner/slug).
  -p, --path DIR     Output destination directory.
      --skip-status  Pull without first requesting kernel status.
  -h, --help         Show this help.

Environment overrides:
  KAGGLE_KERNEL_REF  Default: arcainionprime/taaf-duck-harness-kaggle
  KAGGLE_OUTPUT_DIR  Default: /workspace/duck-harness/results
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -k|--kernel)
            [ "$#" -ge 2 ] || { echo "error: $1 requires a value" >&2; exit 2; }
            KAGGLE_KERNEL_REF="$2"
            shift 2
            ;;
        -p|--path)
            [ "$#" -ge 2 ] || { echo "error: $1 requires a value" >&2; exit 2; }
            KAGGLE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-status)
            CHECK_STATUS=false
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if ! command -v kaggle >/dev/null 2>&1; then
    echo "error: kaggle CLI is not installed or is not on PATH" >&2
    exit 1
fi

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

mkdir -p -- "${KAGGLE_OUTPUT_DIR}"

if [ "${CHECK_STATUS}" = true ]; then
    echo "Checking Kaggle kernel status: ${KAGGLE_KERNEL_REF}"
    if ! kaggle kernels status "${KAGGLE_KERNEL_REF}"; then
        echo "error: unable to read kernel status; verify Kaggle credentials and kernel reference" >&2
        exit 1
    fi
    echo
fi

echo "Pulling Kaggle output: ${KAGGLE_KERNEL_REF}"
echo "Destination: ${KAGGLE_OUTPUT_DIR}"
kaggle kernels output "${KAGGLE_KERNEL_REF}" -p "${KAGGLE_OUTPUT_DIR}"

echo
echo "Kaggle output is available at ${KAGGLE_OUTPUT_DIR}"
