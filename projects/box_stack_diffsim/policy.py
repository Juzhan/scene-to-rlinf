"""Project-local actor-critic using the existing RLinf PPO interface."""

import torch
from torch import Tensor, nn

from .observation import ExecutorObs, ExecutorOutput, ExecutorTargets, PlannerObs, PlannerOutput


class Policy(nn.Module):
    def __init__(self, config):
        super().__init__()
        def network(output_dim):
            layers = []
            for hidden in config.hidden_dims:
                layers.extend((nn.LazyLinear(hidden), nn.ELU()))
            return nn.Sequential(*layers, nn.LazyLinear(output_dim))
        self.register_buffer("named_targets_v1", torch.tensor(1))
        self.actor = network(config.action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(config.action_dim))
        self.value_head = network(1)

    def forward(self, obs: ExecutorObs) -> tuple[ExecutorOutput, Tensor]:
        features = torch.cat((obs.robot.joint_position, obs.robot.joint_velocity,
                              obs.cube_upper.position, obs.cube_lower.position,
                              obs.planner.output, obs.ee_frame.frame_positions[:, 0, :]), -1)
        return self.actor(features), self.value_head(features)


def build_executor(config, torch_dtype):
    return Policy(config).to(dtype=torch_dtype)


class Planner(nn.Module):
    def forward(self, obs: PlannerObs) -> PlannerOutput:
        return obs.cube_upper.target.position


def build_planner():
    return Planner()


def decode_action(action: ExecutorOutput, obs: ExecutorObs) -> ExecutorTargets:
    """Seven arm offsets around the default pose plus one open/close decision."""
    return {'arm': obs.robot.default_joint_position[:, :7] + .5 * action[:, :7],
            'gripper': torch.where(action[:, 7:8] < 0, 0., .04).expand(-1, 2)}
