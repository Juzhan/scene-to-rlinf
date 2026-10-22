#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
arena_dir="$workspace_dir/third_party/IsaacLab-Arena-isaac51"
arena_python="$workspace_dir/.venv-arena51/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PYTHONUNBUFFERED=1
cd "$arena_dir"

case "${1:-view}" in
    view)
        if (( $# > 0 )); then shift; fi
        exec "$arena_python" isaaclab_arena/evaluation/policy_runner.py \
            --policy_type zero_action --num_steps 1000000 "$@" cube_goal_pose
        ;;
    smoke)
        shift
        exec "$arena_python" isaaclab_arena/evaluation/policy_runner.py \
            --headless --policy_type zero_action --num_steps 20 "$@" cube_goal_pose
        ;;
    *)
        # Run another upstream script with this installation's interpreter.
        exec "$arena_python" "$@"
        ;;
esac
