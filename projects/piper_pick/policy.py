"""Planner passes the cube position through; the executor sees joints, the cube and the fingertip centre.

Where to write your own model: ``build_executor`` returns any module whose forward(obs) returns
(action mean [B, 7], value [B, 1]); ``decode_action`` turns the action into the joint targets named in
scene.yaml ``control.output``; ``initial_action`` is the action before the first decision. ``obs`` only
holds the observations selected in scene.yaml (observation.py lists them with shapes and units).
The trained weights ``weights/pick.pt`` (weights/WEIGHTS.json) fix the feature order, READY, STEP and
GRIP below; changing them needs a new training run.
"""
from pathlib import Path

import torch
from torch import Tensor, nn
from .observation import ExecutorObs, ExecutorOutput, ExecutorTargets, PlannerObs, PlannerOutput


class Planner(nn.Module):
    def forward(self, obs: PlannerObs) -> PlannerOutput:
        return obs.cube.position


def build_planner():
    return Planner()


def mlp(sizes, last=None):
    layers = []
    for size in sizes:
        layers.extend((nn.LazyLinear(size), nn.ELU()))
    return nn.Sequential(*layers, *([nn.LazyLinear(last)] if last else []))


class Executor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state = mlp(config.hidden_dims)                      # joints, cube, fingertip centre
        self.actor = mlp(config.hidden_dims, config.action_dim)
        self.actor_logstd = nn.Parameter(torch.full((config.action_dim,), -.5))
        self.value_head = mlp(config.hidden_dims, 1)

    def forward(self, obs: ExecutorObs) -> tuple[ExecutorOutput, Tensor]:
        piper, cube = obs.piper, obs.cube.position
        tcp = obs.tcp_frame.frame_positions[:, 0]
        state = torch.cat((piper.joint_position - piper.joint_position.new_tensor(READY), .1 * piper.joint_velocity,
                           .01 * piper.joint_effort, cube, tcp, 10 * (cube - tcp)), -1)
        features = self.state(state)
        # return action
        return self.actor(features), self.value_head(features)


def build_executor(config, torch_dtype, weights: str | None = None):
    """``weights``: an RLinf actor state dict of this executor (absolute or relative to this project);
    null starts from random initialisation. Checkpoints passed to evaluate/view load on top of it."""
    model = Executor(config)
    if weights:
        path = Path(__file__).parent / weights
        if not path.is_file():
            raise FileNotFoundError(f'{path} is missing; restore weights/pick.pt from git')
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict({key.removeprefix('model.'): value for key, value in state.items()})
    return model.to(dtype=torch_dtype)


# Start pose of every episode (scene.yaml initial_state): fingers pointing down, open, 0.15 m above the
# cube_area centre. (The asset's own default pose holds the gripper sideways at 0.38 m.)
READY = (0., 1.526, -1.295, 1.423, 0., 0., .05, -.05)
# Arm: action i moves joint i by up to STEP rad per decision step (0.02 s) from its measured position.
STEP = .05
# Gripper: one action (the real gripper has a single motor) drives both fingers to mirrored absolute targets
# READY + GRIP * action (m); action -1 closes both, so the fingers always get the same opening. The action is
# clipped to [-1, 0] so the targets stay inside the finger ranges ([0, .05] and [-.05, 0]): an out-of-range target
# keeps the finger drive saturated and closes far faster on Isaac than on mjlab (whose backend clamps targets).
GRIP = (.05, -.05)


def decode_action(action: ExecutorOutput, obs: ExecutorObs) -> ExecutorTargets:
    """Actions 0-5: arm joint increments on the measured position; action 6: shared absolute gripper opening."""
    arm = obs.piper.joint_position[:, :6] + STEP * action[:, :6]
    gripper = action.new_tensor(READY[6:]) + action.new_tensor(GRIP) * action[:, 6:7].clamp(-1., 0.)
    return {'arm': torch.cat((arm, gripper), -1)}


def initial_action(obs: ExecutorObs) -> ExecutorOutput:
    return obs.piper.joint_position.new_zeros(obs.piper.joint_position.shape[0], 7)
