"""Go2 + Piper X whole-body policy after LeggedManip_Lab go2_piper Flat.

The planner is a fixed command sampler: a base velocity command and an end-effector target drawn
uniformly inside the ee_workspace box, both redrawn every planner period and on reset. The executor
is the trained asymmetric actor-critic that drives the 12 leg and 6 arm joints.

Where to write your own model: ``build_executor`` returns any module whose forward(obs) returns
(action mean [B, 18], value [B, 1]); ``decode_action`` turns the action into the joint targets named in
scene.yaml ``control.output``; ``initial_action`` is the action before the first decision; the planner
(``build_planner``) produces the commands the executor reads as ``obs.planner.output``. ``obs`` only holds
the observations selected in scene.yaml (observation.py lists them with shapes and units).
The trained weights ``weights/wholebody.pt`` (weights/WEIGHTS.json) fix the feature order and scales
below; changing them needs a new training run.
"""
from pathlib import Path

import torch
from torch import Tensor, nn
from shared.policy_math import inverse_rotate
from .observation import ExecutorObs, ExecutorOutput, ExecutorTargets, PlannerObs, PlannerOutput

ACTION_SCALE = .25
# vx vy wz in m/s and rad/s, robot base frame.
VELOCITY_LOW, VELOCITY_HIGH = (-1., -.6, -1.), (1., .6, 1.)
STANDING_FRACTION = .1


class CommandSampler(nn.Module):
    """Output [vx, vy, wz, x, y, z]; x y z is the end-effector target in the Piper base frame (m)."""
    def __init__(self):
        super().__init__()
        self.register_buffer('low', torch.tensor(VELOCITY_LOW))
        self.register_buffer('high', torch.tensor(VELOCITY_HIGH))

    def forward(self, obs: PlannerObs) -> PlannerOutput:
        box = obs.ee_workspace
        batch, device = box.position.shape[0], box.position.device
        velocity = self.low + (self.high - self.low) * torch.rand(batch, 3, device=device)
        velocity = velocity * (torch.rand(batch, 1, device=device) >= STANDING_FRACTION)
        # The box follows the Piper base, so its centre in that frame is the authored mount offset.
        centre = inverse_rotate(box.position - obs.piperx.position, box.orientation)
        target = centre + (torch.rand(batch, 3, device=device) - .5) * box.size
        return torch.cat((velocity, target), -1)


def build_planner():
    return CommandSampler()


def actor_features(obs: ExecutorObs) -> Tensor:
    """Deployable inputs: no linear velocity, contact or end-effector pose."""
    robot, arm = obs.robot, obs.piperx
    gravity = inverse_rotate(nn.functional.normalize(obs.simulation.gravity, dim=-1), robot.orientation)
    return torch.cat((.2 * inverse_rotate(robot.angular_velocity, robot.orientation), gravity,
                      robot.joint_position - robot.default_joint_position, arm.joint_position - arm.default_joint_position,
                      .05 * robot.joint_velocity, .05 * arm.joint_velocity,
                      obs.executor.output, obs.planner.output), -1)


def critic_features(obs: ExecutorObs) -> Tensor:
    """Simulation-only extras for the value head (asymmetric actor-critic, as in the reference)."""
    robot, arm = obs.robot, obs.piperx
    tcp = inverse_rotate(arm.tcp.position - arm.position, arm.orientation)
    contact = (obs.feet_contact.contact_force.norm(dim=-1) > 1.).float()
    return torch.cat((2. * inverse_rotate(robot.linear_velocity, robot.orientation), tcp, contact), -1)


class Executor(nn.Module):
    def __init__(self, config):
        super().__init__()
        def network(output_dim):
            layers = []
            for hidden in config.hidden_dims:
                layers.extend((nn.LazyLinear(hidden), nn.ELU()))
            return nn.Sequential(*layers, nn.LazyLinear(output_dim))
        self.actor = network(config.action_dim)
        # Initial noise std 1.0 on the unscaled action, as the reference.
        self.actor_logstd = nn.Parameter(torch.zeros(config.action_dim))
        self.value_head = network(1)

    def forward(self, obs: ExecutorObs) -> tuple[ExecutorOutput, Tensor]:
        features = actor_features(obs)
        return self.actor(features), self.value_head(torch.cat((features, critic_features(obs)), -1))


def build_executor(config, torch_dtype, weights: str | None = None):
    """``weights``: an RLinf actor state dict of this executor (absolute or relative to this project);
    null starts from random initialisation. Checkpoints passed to evaluate/view load on top of it."""
    model = Executor(config)
    if weights:
        path = Path(__file__).parent / weights
        if not path.is_file():
            raise FileNotFoundError(f'{path} is missing; restore weights/wholebody.pt from git')
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict({key.removeprefix('model.'): value for key, value in state.items()})
    return model.to(dtype=torch_dtype)


# Action order = the joint_names of control.output legs and arm in scene.yaml, so decode picks the
# defaults by name. actor_features takes joint tensors as read, in the ASSET.json joint_order that
# both backends present; for Go2 that is LEGS.
LEGS = tuple(f'{leg}_{part}_joint' for part in ('hip', 'thigh', 'calf') for leg in ('FL', 'FR', 'RL', 'RR'))
ARM = tuple(f'joint{index}' for index in range(1, 7))


def _by_name(robot, names):
    order = robot.joint_names()
    return [order.index(name) for name in names]


def _default(robot, names):
    return robot.default_joint_position[:, _by_name(robot, names)]


def decode_action(action: ExecutorOutput, obs: ExecutorObs) -> ExecutorTargets:
    """Unscaled 18-dim action -> joint angles around the default pose (legs first, then joint1..6)."""
    return {'legs': _default(obs.robot, LEGS) + ACTION_SCALE * action[:, :12],
            'arm': _default(obs.piperx, ARM) + ACTION_SCALE * action[:, 12:]}


def initial_action(obs: ExecutorObs) -> ExecutorOutput:
    return obs.robot.joint_position.new_zeros(obs.robot.joint_position.shape[0], 18)
