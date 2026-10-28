#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# An Isaac Lab 3-only install has just .venv-il3; run.py switches interpreters per backend itself.
PYTHON="${RLINF_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  for candidate in "${ROOT}/.venv/bin/python" "${ROOT}/.venv-il3/bin/python"; do
    if [[ -x "$candidate" ]]; then PYTHON="$candidate"; break; fi
  done
fi
if [[ ! -x "$PYTHON" ]]; then
  printf '%s\n' 'Set RLINF_PYTHON to an interpreter with RLinf and the selected simulator installed.' >&2
  exit 1
fi
exec "$PYTHON" "$ROOT/run.py" "$@"