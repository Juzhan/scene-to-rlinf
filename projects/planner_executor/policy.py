"""The policy in two layers (scene.yaml policy.layers): the same Go2 walk to ``destination`` as go2_goal, split.

- ``Planner`` (fixed, every 0.2 s) turns ``destination`` into a goal: it returns a dict, ``goal`` and ``heading``,
  which the executor reads as ``obs.planner.goal`` / ``obs.planner.heading``. Replace it with a planner of your own
  (a path planner, a pretrained model) without touching the executor's interface.
- ``Executor`` (trained, every 0.02 s) walks towards that goal: one action per leg joint, the offset from the
  standing pose ``STANCE`` in units of ``SCALE``.
- ``decode_action`` turns the actions into the robot's joint targets.

``Executor(config, actions)``: ``config`` is train.yaml ``actor.model`` (hidden_dims, and any setting you add there);
``actions`` is the number of joints the policy drives (12). The standing pose and the scale are the policy's own
constants, the same on the real robot.
"""
import torch
from torch import nn
from shared.policy import flat, mlp
from shared.policy_math import inverse_rotate, relative_goal, yaw
from .observation import Action, ExecutorObs, PlannerObs, Targets

# Go2 standing pose (rad), in joint_position order: hips FL FR RL RR, thighs, calves.
STANCE = (.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5)
SCALE = .25  # rad of joint target per unit of action
DOWN = (0., 0., -1.)  # gravity direction, world frame


class Planner(nn.Module):
    """走向哪里、朝向哪里"""
    def forward(self, obs: PlannerObs) -> dict:
        goal = obs.destination
        return {'goal': goal.position,                           # [B, 3] 目标位置（m，环境坐标）
                'heading': yaw(goal.orientation).unsqueeze(-1)}  # [B, 1] 目标朝向（rad，绕 z）


class Executor(nn.Module):
    def __init__(self, config, actions):
        super().__init__()
        self.net = mlp(*config.hidden_dims, actions)

    def forward(self, obs: ExecutorObs) -> Action:
        robot, plan = obs.robot, obs.planner
        relative, error = relative_goal(robot.position, robot.orientation, plan.goal, plan.heading[:, 0])
        distance = torch.linalg.vector_norm(plan.goal - robot.position, dim=-1, keepdim=True)
        down = robot.orientation.new_tensor(DOWN).expand(robot.orientation.shape[0], 3)
        features = flat(inverse_rotate(robot.linear_velocity, robot.orientation),
                        inverse_rotate(robot.angular_velocity, robot.orientation),
                        inverse_rotate(down, robot.orientation),  # gravity direction in the body frame
                        robot.joint_position - robot.joint_position.new_tensor(STANCE), robot.joint_velocity,
                        relative, error, distance, obs.executor.output)  # the last action
        return self.net(features)


def decode_action(action: Action, obs: ExecutorObs) -> Targets:
    """{robot name: [B, joints] targets, in the order of its joint_position}: the standing pose plus the scaled
    offsets (rad). Before the first decision the action is 0: the robot holds the standing pose."""
    return {'robot': action.new_tensor(STANCE) + SCALE * action}
