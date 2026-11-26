#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# run.py picks the interpreter that provides each command's dependency group
# (shared/assembly/runtimes.py); RLINF_PYTHON overrides that choice.
exec "${RLINF_PYTHON:-python3}" "$ROOT/run.py" "$@"
