#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: run.sh RECIPE [training options]" >&2
    exit 2
fi
RECIPE="$1"
shift
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
exec "${PYTHON:-python3}" -m video_proxy.training.launchers.train \
    --recipe "${REPO_ROOT}/configs/${RECIPE}" "$@"
