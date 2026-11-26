"""Go2 + Piper X whole-body policy after LeggedManip_Lab go2_piper Flat.

Two layers (scene.yaml policy): ``command`` is a fixed command sampler: a base velocity command and an
end-effector target drawn uniformly inside the ee_workspace box, both redrawn every 10 s and on reset.
``body`` is the trained asymmetric actor-critic that drives the 12 leg and 6 arm joints; it reads the command as
``obs.command.velocity`` and ``obs.command.ee_target``.

``Body.forward(obs)`` returns the 18 actions; ``value(obs)`` is the value network (named ``value_head`` so RLinf
gives it ``optim.value_lr``), which also reads simulation-only extras. ``decode_action`` turns the actions into joint
targets for every joint of each robot, in the order of its joint_position, around the nominal poses GO2_STANCE and
PIPER_HOME (the policy's constants). ``obs`` only holds the inputs scene.yaml lists (observation.py lists them with
shapes and units). The trained weights ``weights/wholebody.pt`` (weights/WEIGHTS.json) fix the feature order, the
nominal poses and the scales below; changing them needs a new training run.
"""
import torch
from torch import Tensor, nn
from shared.policy import inverse_rotate
from .observation import Action, BodyObs, CommandObs, Targets

ACTION_SCALE = .25
# Nominal poses, in joint_position order: the policy's own constants (the same on the real robots).
GO2_STANCE = (.1, -.1, .1, -.1, .8, .8, 1., 1., -1.5, -1.5, -1.5, -1.5)  # rad: hips FL FR RL RR, thighs, calves
PIPER_HOME = (0., 1.4, -1.2, 0., 0., 0., .05, -.05)  # joint1..6 (rad), gripper_joint1/2 (m, held open)
ACTIONS = 18  # 12 leg joints, then the arm's joint1..6 (decode_action)
# vx vy wz in m/s and rad/s, robot base frame.
VELOCITY_LOW, VELOCITY_HIGH = (-1., -.6, -1.), (1., .6, 1.)
STANDING_FRACTION = .1


class CommandSampler(nn.Module):
    """速度指令与末端目标"""
    def __init__(self):
        super().__init__()
        self.register_buffer('low', torch.tensor(VELOCITY_LOW))
        self.register_buffer('high', torch.tensor(VELOCITY_HIGH))

    def forward(self, obs: CommandObs) -> dict:
        box = obs.ee_workspace
        batch, device = box.position.shape[0], box.position.device
        velocity = self.low + (self.high - self.low) * torch.rand(batch, 3, device=device)
        velocity = velocity * (torch.rand(batch, 1, device=device) >= STANDING_FRACTION)
        # The box follows the Piper base, so its centre in that frame is the authored mount offset.
        centre = inverse_rotate(box.position - obs.piperx.position, box.orientation)
        target = centre + (torch.rand(batch, 3, device=device) - .5) * box.size
        return {'velocity': velocity,  # [B, 3] 机身速度指令 vx、vy（m/s）、wz（rad/s），机身坐标系
                'ee_target': target}   # [B, 3] 末端目标（m），Piper 基座坐标系


def actor_features(obs: BodyObs) -> Tensor:
    """Deployable inputs: no linear velocity, contact or end-effector pose."""
    robot, arm = obs.robot, obs.piperx
    # Gravity direction in the body frame (gravity is the constant (0, 0, -9.81), so only its direction matters).
    gravity = inverse_rotate(robot.orientation.new_tensor((0., 0., -1.)).expand(robot.orientation.shape[0], 3), robot.orientation)
    return torch.cat((.2 * inverse_rotate(robot.angular_velocity, robot.orientation), gravity,
                      robot.joint_position - robot.joint_position.new_tensor(GO2_STANCE),
                      arm.joint_position - arm.joint_position.new_tensor(PIPER_HOME),
                      .05 * robot.joint_velocity, .05 * arm.joint_velocity,
                      obs.body.output, obs.command.velocity, obs.command.ee_target), -1)


def critic_features(obs: BodyObs) -> Tensor:
    """Simulation-only extras for the value head (asymmetric actor-critic, as in the reference)."""
    robot, arm = obs.robot, obs.piperx
    tcp = inverse_rotate(arm.tcp.position - arm.position, arm.orientation)
    contact = (obs.feet_contact.contact_force.norm(dim=-1) > 1.).float()
    return torch.cat((2. * inverse_rotate(robot.linear_velocity, robot.orientation), tcp, contact), -1)


class Body(nn.Module):
    def __init__(self, hidden=(512, 256, 128)):
        super().__init__()
        def network(output_dim):
            layers = []
            for size in hidden:
                layers.extend((nn.LazyLinear(size), nn.ELU()))
            return nn.Sequential(*layers, nn.LazyLinear(output_dim))
        self.actor = network(ACTIONS)
        self.value_head = network(1)

    def forward(self, obs: BodyObs) -> Action:
        return self.actor(actor_features(obs))

    def value(self, obs: BodyObs) -> Tensor:
        return self.value_head(torch.cat((actor_features(obs), critic_features(obs)), -1))


def decode_action(action: Action, obs: BodyObs) -> Targets:
    """18 actions -> joint angles around the nominal poses: the 12 legs (joint_position order), then the arm's
    joint1..6; the gripper is not trained and stays open. Before the first decision the action is 0."""
    legs = action.new_tensor(GO2_STANCE) + ACTION_SCALE * action[:, :12]
    home = action.new_tensor(PIPER_HOME)
    arm = torch.cat((home[:6] + ACTION_SCALE * action[:, 12:], home[6:].expand(action.shape[0], 2)), -1)
    return {'robot': legs, 'piperx': arm}
