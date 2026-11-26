"""The policy: an ordinary PyTorch module that reads ``obs`` and returns actions (scene.yaml ``policy``).

- ``Policy(config, actions)``: ``config`` is train.yaml ``actor.model``; put the network's sizes and your own
  settings there and read them as ``config.<name>``.
- ``Policy.forward(obs)``: ``obs`` holds what scene.yaml lists under ``inputs`` (``obs.robot.joint_position`` ...,
  batch axis first; observation.py has the shapes and units). It returns one action per leg joint: the offset from
  the standing pose ``STANCE``, in units of ``SCALE``. Training adds what PPO needs (action noise, a value network).
- ``decode_action(action, obs)``: the joint targets of every robot, in the order of its joint_position, in real
  units (rad, m, N·m). Scale or convert the actions here (an IK step for end-effector actions, for example).

The standing pose and the scale are the policy's own constants, the same on the real robot. The example walks a Go2
to ``destination``; replace it when the robot or task changes.
"""
import torch
from shared.policy import flat, inverse_rotate, mlp, relative_goal, yaw
from torch import nn
from .observation import Action, PolicyObs, Targets

# Go2 standing pose (rad), in joint_position order: hips FL FR RL RR, thighs, calves.
STANCE = (.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5)
SCALE = .25  # rad of joint target per unit of action
DOWN = (0., 0., -1.)  # gravity direction, world frame


class Policy(nn.Module):
    def __init__(self, config, actions):
        """``config``: train.yaml ``actor.model`` (hidden_dims, and any setting you add there);
        ``actions``: the robot's joints (12)."""
        super().__init__()
        self.net = mlp(*config.hidden_dims, actions)

    def forward(self, obs: PolicyObs) -> Action:
        robot, goal = obs.robot, obs.destination
        relative, error = relative_goal(robot.position, robot.orientation, goal.position, yaw(goal.orientation))
        distance = torch.linalg.vector_norm(goal.position - robot.position, dim=-1, keepdim=True)
        down = robot.orientation.new_tensor(DOWN).expand(robot.orientation.shape[0], 3)
        features = flat(inverse_rotate(robot.linear_velocity, robot.orientation),
                        inverse_rotate(robot.angular_velocity, robot.orientation),
                        inverse_rotate(down, robot.orientation),  # gravity direction in the body frame
                        robot.joint_position - robot.joint_position.new_tensor(STANCE), robot.joint_velocity,
                        relative, error, distance, obs.policy.output)  # the last action
        return self.net(features)


def decode_action(action: Action, obs: PolicyObs) -> Targets:
    """{robot name: [B, joints] targets, in the order of its joint_position}: the standing pose plus the scaled
    offsets (rad). Before the first decision the action is 0: the robot holds the standing pose."""
    return {'robot': action.new_tensor(STANCE) + SCALE * action}
