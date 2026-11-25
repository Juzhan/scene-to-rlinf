"""The policy: one layer reading the arm, the cube and the fingertip centre (scene.yaml policy).

``Policy.forward(obs)`` returns 7 actions; ``value(obs)`` is the value network PPO trains with (named ``value_head``
so RLinf gives it ``optim.value_lr``). ``decode_action`` turns the actions into joint targets for every joint of the
robot, in the order of its joint_position; an action of 0 holds the arm where it is and the gripper open. ``obs``
only holds the inputs scene.yaml lists (observation.py lists them with shapes and units).
The trained weights ``weights/pick.pt`` (weights/WEIGHTS.json) fix the feature order, REFERENCE, STEP, OPEN and
GRIP below; changing them needs a new training run.
"""
import torch
from torch import Tensor, nn
from .observation import Action, PolicyObs, Targets


def mlp(sizes, last=None):
    layers = []
    for size in sizes:
        layers.extend((nn.LazyLinear(size), nn.ELU()))
    return nn.Sequential(*layers, *([nn.LazyLinear(last)] if last else []))


class Policy(nn.Module):
    def __init__(self, hidden=(256, 128)):
        super().__init__()
        self.state = mlp(hidden)                     # joints, cube, fingertip centre
        self.actor = mlp(hidden, ACTIONS)
        self.value_head = mlp(hidden, 1)

    def features(self, obs: PolicyObs) -> Tensor:
        piper, cube = obs.piper, obs.cube.position
        tcp = obs.tcp_frame.frame_positions[:, 0]
        return self.state(torch.cat((piper.joint_position - piper.joint_position.new_tensor(REFERENCE), .1 * piper.joint_velocity,
                                     .01 * piper.joint_effort, cube, tcp, 10 * (cube - tcp)), -1))

    def forward(self, obs: PolicyObs) -> Action:
        return self.actor(self.features(obs))

    def value(self, obs: PolicyObs) -> Tensor:
        return self.value_head(self.features(obs))


# The pose the joint features are measured from (a constant of the policy, fixed by the weights): the arm pointing
# down 0.15 m above the cube_area centre, fingers open.
REFERENCE = (0., 1.526, -1.295, 1.423, 0., 0., .05, -.05)
ACTIONS = 7  # six arm joint steps and one gripper opening (decode_action)
# Arm: action i moves joint i by up to STEP rad per decision step (0.02 s) from its measured position.
STEP = .05
# Gripper: one action (the real gripper has a single motor) drives both fingers to mirrored absolute targets
# OPEN + GRIP * action (m); action -1 closes both, so the fingers always get the same opening. The action is
# clipped to [-1, 0] so the targets stay inside the finger ranges ([0, .05] and [-.05, 0]): an out-of-range target
# keeps the finger drive saturated and closes far faster on Isaac than on mjlab (whose backend clamps targets).
OPEN = (.05, -.05)  # finger targets fully open (m)
GRIP = (.05, -.05)


def decode_action(action: Action, obs: PolicyObs) -> Targets:
    """Actions 0-5: arm joint increments on the measured position; action 6: shared absolute gripper opening."""
    arm = obs.piper.joint_position[:, :6] + STEP * action[:, :6]
    gripper = action.new_tensor(OPEN) + action.new_tensor(GRIP) * action[:, 6:7].clamp(-1., 0.)
    return {'piper': torch.cat((arm, gripper), -1)}

