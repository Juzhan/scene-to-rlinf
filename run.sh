#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${RLINF_PYTHON:-${ROOT}/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Set RLINF_PYTHON to an interpreter with RLinf and the selected simulator installed.' >&2
  exit 1
fi
exec "$PYTHON" "$ROOT/run.py" "$@"