"""Chase a fleeing virtual ball; dense approach/facing reward and a one-shot tag success.

Mirrors the chase half of go2-supermimic's flat task (chase_approach, facing_target, upright, joint
deviation, action rate, outside arena, fall termination). The ball is an abstract point (no body, no
contact) that the Task moves every decision step (``set``, computed by ``move_ball``): it flees from the
robot with Ornstein-Uhlenbeck jukes, bounded by a wall, as the reference's chase ball does. The move
takes effect after resets, before the next observation; a freshly reset ball keeps its sampled spot. The reference's ball bursts away after
a tag; here a tag ends the episode. Thrown dodge balls/bars, LiDAR and the height scan are not
reproduced (see the project README).
"""
import math
import torch
from torch import Tensor
from .observation import SceneInfo, TaskResult
from shared.policy_math import inverse_rotate, yaw

ARENA_RADIUS = 4.0        # m; the terrain block is 8 m wide
TAG_RADIUS = .2           # horizontal head-to-ball distance that counts as a tag (m), as in the reference
TAG_SIGMA = .3            # Gaussian width of the dense tag term (m)
# Tagging ends the episode, so the dense tag term cannot be farmed by hovering next to the ball.
SUCCESS_BONUS = 20 / .02  # 20 return, above the remaining dense return of a 6 s episode
FALL_COS = math.cos(math.radians(70.))   # projected-gravity z below -cos(70 deg) is a fall
# Ball motion: fixed values from the start of the reference curriculum (target/generator.py).
BALL_ARENA = 3.5          # m; the ball stays this close to the environment origin, inside ARENA_RADIUS
FLEE_RATIO = .85          # flee speed as a fraction of the robot's planar speed
FLEE_MIN_SPEED = .5       # m/s; flee speed when the robot stands still
FLEE_GAIN = .5            # fraction of the flee speed applied
SPEED_CAP = 1.25          # m/s
OU_THETA = 1.5            # 1/s; mean reversion of the velocity toward the flee velocity
OU_SIGMA = 4.             # m/s^2 per sqrt(s); random sideways jukes


def move_ball(position: Tensor, velocity: Tensor, robot, dt: Tensor) -> Tensor:
    """Next ball position ['B', 3]: flee from the robot plus Ornstein-Uhlenbeck noise, held inside BALL_ARENA.

    No hidden state: the random part is the previous velocity (``ball.linear_velocity``, zero after a
    reset) minus the current flee velocity; it decays at OU_THETA with OU_SIGMA noise, like the
    reference's OU jukes, and is bounded so the speed stays within SPEED_CAP. Height stays as sampled.
    ``dt`` is the decision step ['B', 1] (s).
    """
    xy = position[:, :2]
    away = xy - robot.position[:, :2]
    away = away / torch.linalg.vector_norm(away, dim=-1, keepdim=True).clamp(min=1e-6)
    flee_speed = (FLEE_RATIO * torch.linalg.vector_norm(robot.linear_velocity[:, :2], dim=-1)).clamp(min=FLEE_MIN_SPEED)
    flee = away * (FLEE_GAIN * flee_speed)[:, None]
    jitter = (velocity[:, :2] - flee) * (1. - OU_THETA * dt) + torch.randn_like(flee) * OU_SIGMA * dt ** .5
    # Bound the random part, not the sum: capping the sum would leave a pull toward the robot in the
    # next step's velocity - flee.
    room = (SPEED_CAP - torch.linalg.vector_norm(flee, dim=-1, keepdim=True)).clamp(min=0.)
    jitter = jitter * (room / torch.linalg.vector_norm(jitter, dim=-1, keepdim=True).clamp(min=1e-6)).clamp(max=1.)
    move = flee + jitter
    # Wall: drop the outward part near it, push back in beyond it.
    radius = torch.linalg.vector_norm(xy, dim=-1, keepdim=True).clamp(min=1e-6)
    radial = xy / radius
    outward = (move * radial).sum(-1, keepdim=True).clamp(min=0.)
    move = move - (radius > BALL_ARENA - .3) * outward * radial - (radius > BALL_ARENA) * .5 * radial
    speed = torch.linalg.vector_norm(move, dim=-1, keepdim=True).clamp(min=1e-6)
    move = move * (SPEED_CAP / speed).clamp(max=1.)
    return torch.cat((xy + move * dt, position[:, 2:]), -1)


def evaluate(obs: SceneInfo) -> TaskResult:
    robot, ball = obs.robot, obs.ball.position
    q = robot.joint_position - robot.default_joint_position
    heading = yaw(robot.orientation)
    c, s = heading.cos(), heading.sin()
    offset = ball - robot.position
    distance = torch.linalg.vector_norm(offset[:, :2], dim=-1).clamp(min=1e-6)
    direction = offset[:, :2] / distance[:, None]
    speed_toward = (robot.linear_velocity[:, :2] * direction).sum(-1).clamp(-5., 5.)
    facing = .5 * (1. + c * direction[:, 0] + s * direction[:, 1])
    gravity = inverse_rotate(robot.joint_position.new_tensor((0., 0., -1.)).expand_as(ball), robot.orientation)
    upright = torch.exp(-gravity[:, :2].square().sum(-1) / .4**2)
    angular = inverse_rotate(robot.angular_velocity, robot.orientation)
    tilt_rate = angular[:, :2].square().sum(-1)
    # Horizontal, as in the reference: the virtual ball has no height to touch.
    head = obs.head.frame_positions[:, 0]
    head_distance = torch.linalg.vector_norm(head[:, :2] - ball[:, :2], dim=-1)
    tag = torch.exp(-(head_distance / TAG_SIGMA).square())
    fallen = (gravity[:, 2] > -FALL_COS) | (robot.position[:, 2] < -.2)
    outside = (torch.linalg.vector_norm(robot.position[:, :2], dim=-1) - ARENA_RADIUS).clamp(min=0.)
    success = (head_distance < TAG_RADIUS) & ~fallen
    change = (obs.executor.output - obs.executor.previous_output).square().sum(-1)
    reward = (1.5 * speed_toward + .5 * facing + 1.5 * tag + upright - 2. * outside
              - .05 * tilt_rate - .1 * q.square().sum(-1) - .05 * change
              - 2e-5 * robot.joint_effort.square().sum(-1)
              - 10. * (.15 - robot.position[:, 2]).clamp(min=0.)
              - 200. * fallen + SUCCESS_BONUS * success)
    return {'reward': reward, 'success': success, 'failure': fallen,
            'metrics': {'distance_m': distance, 'head_distance_m': head_distance,
                        'speed_toward_mps': speed_toward, 'tagged': success, 'fell': fallen},
            'set': {'ball': move_ball(ball, obs.ball.linear_velocity, robot, obs.simulation.step_dt)}}
