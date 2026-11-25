"""Chase policy: the released Go2-SuperMimic actor as the one policy layer (scene.yaml policy).

``Policy`` reproduces the actor of Zhengtianjin/go2-supermimic (MIT, commit 054a13c) and loads its
flat-ground weights ``weights/flat.pt`` (see weights/WEIGHTS.json). The weights fix everything below:
feature order, the 5-frame history, the per-leg joint order, stance and action scale. Changing any of
them needs a new training run, not an edit here. The scene drives the robot with the matching
``go2(actuators: supermimic)`` motors; with other motors these weights do not walk.

``forward(obs)`` returns the 12 actions; ``value(obs)`` is the value network PPO fine-tunes with (the released
checkpoint has none; it is named ``value_head`` so RLinf gives it ``optim.value_lr``). ``decode_action`` turns the
actions into joint targets for every joint of the robot, in the order of its joint_position. ``obs`` only holds the
inputs scene.yaml lists (observation.py lists them with shapes and units).
"""
from pathlib import Path

import torch
from torch import Tensor, nn
from shared.policy_math import inverse_rotate, yaw
from .observation import Action, PolicyObs, Targets

# The actor orders legs FL, FR, RL, RR with hip, thigh, calf per leg; the robot reports and takes joints
# as hips, thighs, calves (ASSET.json joint_order).
ASSET_TO_ACTOR = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
ACTOR_TO_ASSET = [ASSET_TO_ACTOR.index(joint) for joint in range(12)]
STANCE = (.1, .9, -1.8, -.1, .9, -1.8, .1, .9, -1.8, -.1, .9, -1.8)  # rad, actor order
SCALE = (.3349, .3349, .2853) * 4  # rad per action unit, actor order


class RunningNorm(nn.Module):
    """Observation normaliser of the reference; frozen statistics loaded with the weights."""

    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size))
        self.register_buffer('var', torch.ones(size))
        self.register_buffer('count', torch.tensor(1e-4))

    def forward(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10., 10.)


def mlp(size, hidden, last):
    layers = []
    for width in hidden:
        layers += [nn.Linear(size, width), nn.ELU()]
        size = width
    return nn.Sequential(*layers, nn.Linear(size, last))


def yaw_frame(vector, heading):
    c, s = heading.cos(), heading.sin()
    return torch.stack((c * vector[..., 0] + s * vector[..., 1],
                        -s * vector[..., 0] + c * vector[..., 1], vector[..., 2]), -1)


def features(obs: PolicyObs) -> Tensor:
    """350 inputs: six terms, each the 5 frames oldest -> newest (scene.yaml policy history)."""
    robot = obs.robot
    orientation = robot.orientation  # [B, 5, 4]
    stance = orientation.new_tensor(STANCE)
    gravity = inverse_rotate(orientation.new_tensor((0., 0., -1.)).expand(*orientation.shape[:-1], 3), orientation)
    heading = yaw(orientation)
    # Target slot of the reference: ball offset and velocity in the yaw frame, zero uncertainty,
    # visible, age 0, a point (not a bar), active. The second (dodge) slot is empty.
    offset = yaw_frame(obs.ball.position - robot.position, heading)
    velocity = yaw_frame(obs.ball.linear_velocity, heading)
    one, zero = torch.ones_like(offset[..., :1]), torch.zeros_like(offset[..., :1])
    slot = torch.cat((offset, velocity, torch.zeros_like(offset), one, zero, zero, zero, one), -1)
    terms = (inverse_rotate(robot.angular_velocity, orientation), gravity,
             robot.joint_position[..., ASSET_TO_ACTOR] - stance, robot.joint_velocity[..., ASSET_TO_ACTOR],
             obs.policy.output, torch.cat((slot, torch.zeros_like(slot)), -1))
    return torch.cat([term.flatten(1) for term in terms], -1)


class Actor(nn.Module):
    """The reference actor; its state dict is the ``actor`` entry of the weights file."""

    def __init__(self, inputs=350, hidden=(512, 256, 128), actions=12):
        super().__init__()
        self.norm = RunningNorm(inputs)
        self.net = mlp(inputs, hidden, actions)


class Policy(nn.Module):
    def __init__(self, actor_file: str | None = 'weights/flat.pt', value_hidden=(512, 256, 128)):
        """``actor_file``: a go2-supermimic actor file (absolute or relative to this project); null starts from
        random initialisation. Checkpoints passed to evaluate/view load on top of it."""
        super().__init__()
        self.actor = Actor()
        self.value_head = mlp(350, value_hidden, 1)
        if actor_file:
            path = Path(__file__).parent / actor_file
            if not path.is_file():
                raise FileNotFoundError(f'{path} is missing; restore weights/flat.pt from git')
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint.get('obs_dim') != 350:
                raise ValueError(f'{path}: expected a 350-input go2-supermimic actor, got {checkpoint.get("obs_dim")}')
            self.actor.load_state_dict(checkpoint['actor'])

    def forward(self, obs: PolicyObs) -> Action:
        return self.actor.net(self.actor.norm(features(obs)))

    def value(self, obs: PolicyObs) -> Tensor:
        return self.value_head(self.actor.norm(features(obs)))


def decode_action(action: Action, obs: PolicyObs) -> Targets:
    """Joint-position targets around the stance, computed in actor order and given in the robot's."""
    return {'robot': (action.new_tensor(STANCE) + action.new_tensor(SCALE) * action)[:, ACTOR_TO_ASSET]}

