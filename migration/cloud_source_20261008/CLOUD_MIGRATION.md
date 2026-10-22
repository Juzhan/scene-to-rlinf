# Cloud Migration

## Upload and extract

Upload `project.tar.zst`, `SHA256SUMS`, this guide and the package inventory JSON
files to a persistent cloud directory. No historical training checkpoints or
replay are included. Install `zstd` if the container lacks it.
Run from the upload directory:

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar --zstd -xf project.tar.zst
cd rlinf-workspace
```

The archive has a single `rlinf-workspace/` root. Do not extract over an active
experiment.

## Contents

- All projects, including Isaac visual Franka SAC, native Franka Lift, Go2 and
  Go2 platforms, MuJoCo Franka/stack/stack v2, and other workspace tasks.
- All workspace assets and third-party working source trees, including local
  RLinf modifications. Git metadata and generated build/cache files are excluded.
- Local IsaacLab source at `third_party/IsaacLab` (v2.3.2), not Isaac Sim binaries.
- ResNet10 weights and dereferenced DINOv2-small weights at `models/dinov2-small`.
- In the cloud copy only, stack-v2 actor warm-start is disabled (null) and its
  preflight accepts fresh initialization. The original local training source and
  configuration remain unchanged. All task/reward/camera/action settings stay
  the same; cloud training is a new experiment, not continuation of the old one.

No virtual environments, Conda installation, Isaac Sim binaries, system CUDA,
historical logs, videos, learned policies, replay or training checkpoints are included.
Go2 has source/configuration but no trained checkpoint in this bundle. Some
Isaac tasks still use official network-hosted assets; Go2 and native Lift are
not certified fully offline. The packaged Panda asset cache is local.

## Rebuild environments

Do not use the local `.venv` symlinks on the cloud machine. Rebuild separate
MuJoCo and Isaac environments; the included package inventory JSON files are
reference evidence, not automatically installable or guaranteed-compatible
lockfiles. Inspect the container OS, NVIDIA driver, Python and GPU first:

```bash
nvidia-smi
python --version
df -h . /dev/shm
```

The local MuJoCo setup uses Python 3.11, MuJoCo 2.3.7, Gym 0.26, NumPy 1.26.4,
Ray 2.48 and Torch 2.7.0+cu128. See `third_party/serl-frankasim` and RLinf's
installation scripts for additional dependencies. Install the bundled FrankaSim
source rather than fetching an unmodified replacement. Use an environment with
CUDA-enabled Torch, working NVIDIA EGL libraries and graphics capabilities:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export RLINF_PYTHON="$PWD/.venv-frankasim/bin/python"
bash run.sh check frankasim_stack_v2
```

For Isaac, install a compatible Isaac Sim runtime and install the bundled
IsaacLab source into that environment. Merely unpacking `third_party/IsaacLab`
does not install Isaac Sim. Read its runtime/version requirements before
installation. A cloud GPU exposed only for compute may not support EGL/Vulkan
camera rendering until the provider enables NVIDIA graphics capabilities.
The existing container need not and may not support running Docker inside it.

After rebuilding Isaac dependencies:

```bash
export RLINF_PYTHON="$PWD/.venv/bin/python"
bash run.sh check franka_isaac
bash run.sh check franka_lift
bash run.sh check go2
bash run.sh check go2 --config platforms
```

`check` does not launch simulation, prove GPU rendering, or certify training.
Next run a bounded smoke test, then two actual training updates with checkpoint
verification before a long job. Start conservatively on one GPU and measure
sampling/update throughput and CPU/RAM/VRAM before increasing parallelism.
Preserve the two actual 720p cameras; network inputs are resized separately.

For old DINO recipes override the local absolute model path at launch:

```bash
bash run.sh check my_robot --config dino \
  "actor.model.visual_checkpoint=$PWD/models/dinov2-small"
```

## Fresh experiment

ResNet10 and DINO are public pretrained visual backbones, not historical RL
training checkpoints. They remain included because the policies require them.
The cloud stack-v2 recipe starts a new actor/Q/optimizer/replay rather than
inheriting the v1 learned actor. Its results are not a continuation or directly
equivalent initialization to the local v2 experiment.

Use new output directories and experiment records. Existing experiment drivers
may enforce actor inheritance or frozen-plan assumptions; for a fresh cloud v2
run use `bash run.sh train frankasim_stack_v2` after successful preflight/smoke
and a two-update training test, not an old frozen experiment plan.
The package-only adaptation is recorded in `tools/cloud_fresh_stack.patch` and
applied by the packaging script to temporary copies, not local training files.
Nothing in this bundle automatically starts training or changes the local jobs.
Do not infer task success from reward increase or a successful migration.