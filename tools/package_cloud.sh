#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${ROOT}/migration/cloud_source_20261008}"
ISAACLAB="${ISAACLAB_SOURCE:-/home/jz/repos/IsaacLab}"
DINO="${DINO_SOURCE:-/home/jz/.cache/huggingface/hub/models--facebook--dinov2-small/snapshots/ed25f3a31f01632728cabb09d1542f84ab7b0056}"

for executable in tar zstd sha256sum patch; do
    command -v "$executable" >/dev/null
done
for required in "$ISAACLAB/source/isaaclab" "$DINO/model.safetensors" "$DINO/config.json"; do
    [[ -e "$required" ]] || { printf 'Missing: %s\n' "$required" >&2; exit 1; }
done
mkdir -p "$DEST"
DEST="$(cd -- "$DEST" && pwd)"
for package in project; do
    [[ ! -e "$DEST/$package.tar.zst" ]] || { printf 'Refusing to overwrite %s\n' "$package" >&2; exit 1; }
done
STAGING="$(mktemp -d)"
trap 'rm -rf -- "$STAGING"' EXIT
mkdir -p "$STAGING/cloud_overrides/projects/frankasim_stack_v2"
cp "$ROOT/projects/frankasim_stack_v2/project.py" "$ROOT/projects/frankasim_stack_v2/train.yaml" "$STAGING/cloud_overrides/projects/frankasim_stack_v2/"
patch --batch --fuzz=0 -d "$STAGING/cloud_overrides" -p1 < "$ROOT/tools/cloud_fresh_stack.patch"

"$ROOT/.venv-frankasim/bin/python" - "$STAGING/cloud_overrides" <<'PY'
import runpy
import sys
from pathlib import Path
from unittest.mock import patch
from omegaconf import OmegaConf
sys.path.insert(0, str(Path.cwd()))
config = OmegaConf.load(Path(sys.argv[1]) / "projects/frankasim_stack_v2/train.yaml")
assert config.actor.model.initial_actor_checkpoint is None
assert config.rollout.model.initial_actor_checkpoint is None
preflight = runpy.run_path(str(Path(sys.argv[1]) / "projects/frankasim_stack_v2/project.py"))["preflight"]
with patch("projects.frankasim_stack.project.preflight") as runtime_check:
    preflight(config)
    runtime_check.assert_called_once_with(config)
config.actor.model.initial_actor_checkpoint = "/missing/cloud_checkpoint.pt"
with patch("projects.frankasim_stack.project.preflight"):
    try:
        preflight(config)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Missing explicit warm-start must still fail")
print("PASS cloud copy: fresh initialization and missing-checkpoint guard")
PY

cp "$ROOT/tools/CLOUD_MIGRATION.md" "$DEST/CLOUD_MIGRATION.md"
for environment in .venv .venv-frankasim; do
    "$ROOT/$environment/bin/python" -m pip list --format=json > "$DEST/${environment#.}-packages.json"
done
for repository in RLinf serl-frankasim agx_arm_sim curobo; do
    git -C "$ROOT/third_party/$repository" rev-parse HEAD > "$DEST/$repository-commit.txt"
    git -C "$ROOT/third_party/$repository" diff HEAD --binary > "$DEST/$repository-local-changes.patch"
done
git -C "$ISAACLAB" rev-parse HEAD > "$DEST/IsaacLab-commit.txt"
git -C "$ISAACLAB" diff HEAD --binary > "$DEST/IsaacLab-local-changes.patch"
sha256sum "$DINO/model.safetensors" "$DINO/config.json" > "$DEST/source-models.sha256"

tar --dereference --use-compress-program='zstd -T4 -3' \
    --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='.pytest_cache' --exclude='.ruff_cache' --exclude='*.egg-info' \
    --exclude='build' --exclude='.env' --exclude='.env.*' \
    --exclude='rlinf-workspace/projects/frankasim_stack_v2/project.py' \
    --exclude='rlinf-workspace/projects/frankasim_stack_v2/train.yaml' \
    --transform='s,^IsaacLab\(/\|$\),rlinf-workspace/third_party/IsaacLab\1,' \
    --transform='s,^cloud_overrides\(/\|$\),rlinf-workspace\1,' \
    --transform="s,^$(basename "$DINO")\\(/\\|$\\),rlinf-workspace/models/dinov2-small\\1," \
    -cf "$DEST/project.tar.zst" \
    -C "$(dirname "$ROOT")" \
    rlinf-workspace/README.md rlinf-workspace/pyproject.toml \
    rlinf-workspace/run.py rlinf-workspace/run.sh \
    rlinf-workspace/assets rlinf-workspace/configs rlinf-workspace/projects \
    rlinf-workspace/shared rlinf-workspace/tests rlinf-workspace/tools \
    rlinf-workspace/third_party \
    -C "$(dirname "$ISAACLAB")" "$(basename "$ISAACLAB")" \
    -C "$(dirname "$DINO")" "$(basename "$DINO")" \
    -C "$STAGING" cloud_overrides

for package in "$DEST"/*.tar.zst; do
    zstd -t "$package"
    tar --use-compress-program=zstd -tf "$package" > "$package.contents.txt"
done
for required in projects/franka_isaac/train.yaml projects/franka_lift/train.yaml projects/go2/train.yaml projects/go2/platforms.yaml projects/frankasim_stack_v2/train.yaml projects/frankasim_stack_v2/project.py assets/franka_panda/panda_instanceable.usd third_party/RLinf/rlinf/algorithms/utils.py third_party/IsaacLab/isaaclab.sh models/dinov2-small/model.safetensors; do
    grep -Fx "rlinf-workspace/$required" "$DEST/project.tar.zst.contents.txt" >/dev/null
done
if grep -E '^rlinf-workspace/(outputs|\.venv[^/]*)(/|$)' "$DEST/project.tar.zst.contents.txt"; then
    printf 'Unexpected historical outputs or environment files in archive\n' >&2
    exit 1
fi
cd "$DEST"
sha256sum ./*.tar.zst > SHA256SUMS
sha256sum -c SHA256SUMS
du -h ./*.tar.zst
printf 'Packages ready: %s\n' "$DEST"