#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: run.sh {seg|prog|relation} {paper|cot} [training options]" >&2
    exit 2
fi
case "$1" in seg|prog|relation) ;; *) exit 2 ;; esac
case "$2" in paper|cot) ;; *) exit 2 ;; esac
RECIPE="$2/$1_teacher.yaml"
shift 2
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../../training/recipes/run.sh" "${RECIPE}" "$@"
