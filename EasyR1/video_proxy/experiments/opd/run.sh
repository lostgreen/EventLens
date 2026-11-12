#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: run.sh {4b|8b} [training options]" >&2
    exit 2
fi
case "$1" in 4b|8b) ;; *) exit 2 ;; esac
RECIPE="paper/student_$1.yaml"
shift
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../training/recipes/run.sh" "${RECIPE}" "$@"
