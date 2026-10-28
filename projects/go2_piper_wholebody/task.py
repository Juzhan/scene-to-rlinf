"""Reward and termination terms of LeggedManip_Lab go2_piper Flat; command = planner.output [vx vy wz | ee x y z].

Where to write the task: ``Task.evaluate(obs)`` runs once per decision step for all environments and
returns [B] tensors ``reward`` (multiplied by step_dt when summed), ``failure`` (ends the episode; there
is no success here) and ``metrics``. ``obs`` (SceneInfo in observation.py) reads every entity, including
ground truth the policy does not see. ``reset(env_ids)`` clears the per-environment state.
"""
import torch
from .observation import SceneInfo, TaskResult
from shared.assembly.abstract import quaternion_multiply
from shared.policy_math import inverse_rotate

FEET = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')  # also the order of feet_contact bodies
# Piper base frame -> desired tcp: approach axis (gripper z) forward, gripper x down; the reference's
# identity end-effector command with its end_effector frame rotated -90 deg about y from link6.
TCP_COMMAND_ORIENTATION = (.7071068, 0., .7071068, 0.)
# Base above the lowest foot centre (m), so the term holds on uneven ground. The default stance is
# ~0.296; the reference's 0.28 absolute height rewarded a crouch.
BASE_HEIGHT = .30
AIR_TIME_THRESHOLD, MAX_AIR_TIME = .5, 1.

WEIGHTS = {
    'ee_position': 3., 'ee_orientation': -1.5, 'track_lin_vel_xy': 3., 'track_ang_vel_z': 1.5, 'base_height': 1.,
    'lin_vel_z': -2., 'ang_vel_xy': -.05, 'torques': -1e-5, 'hip_torque_max': -8e-5, 'thigh_torque_max': -8e-5,
    'calf_torque_max': -2e-5, 'joint_acc': -2.5e-7, 'action_rate': -.01, 'joint_power': -2e-5, 'feet_slide': -.1,
    'hip_deviation': -.1, 'leg_deviation': -.02, 'arm_deviation': -.1, 'joint_mirror': -.15, 'feet_air_time': .5,
    'feet_long_air': -.5, 'air_time_variance': -1., 'flat_orientation': -1.,
}


class Task:
    """Per-foot air/contact timers and previous readings are episode state, cleared by reset(env_ids)."""
    def __init__(self):
        self.state, self.index = {}, None

    def reset(self, env_ids):
        for value in self.state.values():
            value[slice(None) if env_ids is None else env_ids] = 0

    def _state(self, name, like):
        if name not in self.state:
            self.state[name] = torch.zeros_like(like)
        return self.state[name]

    def _indices(self, obs):
        legs, arm = obs.robot.joint_names(), obs.piperx.joint_names()
        pick = lambda names, keys: torch.tensor([names.index(k) for k in keys], device=obs.robot.joint_position.device)
        side = lambda leg: [f'{leg}_{part}_joint' for part in ('hip', 'thigh', 'calf')]
        return {'hip': pick(legs, [n for n in legs if n.endswith('_hip_joint')]),
                'thigh': pick(legs, [n for n in legs if n.endswith('_thigh_joint')]),
                'calf': pick(legs, [n for n in legs if n.endswith('_calf_joint')]),
                'mirror': [(pick(legs, side(a)), pick(legs, side(b))) for a, b in (('FR', 'RL'), ('FL', 'RR'))],
                'arm': pick(arm, [f'joint{i}' for i in range(1, 7)])}

    def evaluate(self, obs: SceneInfo) -> TaskResult:
        robot, arm, dt = obs.robot, obs.piperx, obs.simulation.step_dt[:, 0]
        if self.index is None:
            self.index = self._indices(obs)
        index, command = self.index, obs.planner.output
        q = robot.orientation
        lin, ang = inverse_rotate(robot.linear_velocity, q), inverse_rotate(robot.angular_velocity, q)
        gravity = inverse_rotate(torch.nn.functional.normalize(obs.simulation.gravity, dim=-1), q)
        upright = -gravity[:, 2]
        tilt = torch.clamp(upright, 0, .7) / .7

        # End effector against the command in the moving Piper base frame.
        tcp_position, tcp_orientation = arm.frame_pose('tcp')
        ee = inverse_rotate(tcp_position - arm.position, arm.orientation)
        ee_error = (ee - command[:, 3:]).square().sum(-1)
        desired = quaternion_multiply(q, q.new_tensor(TCP_COMMAND_ORIENTATION).expand_as(q))
        ee_angle = 2 * torch.acos((desired * tcp_orientation).sum(-1).abs().clamp(max=1.))

        # Joint quantities over the 12 leg and 6 arm joints the policy drives.
        arm_position = arm.joint_position[:, index['arm']]
        arm_default = arm.default_joint_position[:, index['arm']]
        effort = torch.cat((robot.joint_effort, arm.joint_effort[:, index['arm']]), -1)
        velocity = torch.cat((robot.joint_velocity, arm.joint_velocity[:, index['arm']]), -1)
        previous_velocity = self._state('joint_velocity', velocity)
        seen = self._state('seen', dt)
        acceleration = torch.where(seen[:, None] > 0, (velocity - previous_velocity) / dt[:, None], 0.)
        offset = robot.joint_position - robot.default_joint_position
        torque_max = {part: effort[:, index[part]].abs().amax(-1).square() for part in ('hip', 'thigh', 'calf')}
        mirror = sum((robot.joint_position[:, a] - robot.joint_position[:, b]).square().sum(-1)
                     for a, b in index['mirror']) / len(index['mirror'])

        # Feet: contact from the sensor window, air/contact timers updated once per decision step.
        force = obs.feet_contact.contact_force_samples.norm(dim=-1).amax(1)
        contact = force > 1.
        air, contact_time = self._state('air', force), self._state('contact_time', force)
        last_air, last_contact = self._state('last_air', force), self._state('last_contact', force)
        first_contact = contact & (air > 0)
        last_air.copy_(torch.where(first_contact, air + dt[:, None], last_air))
        last_contact.copy_(torch.where(~contact & (contact_time > 0), contact_time + dt[:, None], last_contact))
        air.copy_(torch.where(contact, 0., air + dt[:, None]))
        contact_time.copy_(torch.where(contact, contact_time + dt[:, None], 0.))
        feet = torch.stack([robot.frame_pose(name)[0] for name in FEET], 1)
        previous_feet = self._state('feet', feet)
        foot_velocity = torch.where(seen[:, None, None] > 0, (feet - previous_feet) / dt[:, None, None], 0.)
        relative = foot_velocity - robot.linear_velocity[:, None]
        foot_lateral = inverse_rotate(relative, q[:, None].expand(-1, len(FEET), -1))[..., :2].norm(dim=-1)
        moving = command[:, :2].norm(dim=-1) > .1
        previous_velocity.copy_(velocity)
        previous_feet.copy_(feet)
        seen.fill_(1)
        height = robot.position[:, 2] - feet[..., 2].amin(1)

        terms = {
            'ee_position': torch.exp(-ee_error / .1),
            'ee_orientation': ee_angle,
            'track_lin_vel_xy': torch.exp(-(command[:, :2] - lin[:, :2]).square().sum(-1) / .25),
            'track_ang_vel_z': torch.exp(-(command[:, 2] - ang[:, 2]).square() / .25),
            'base_height': torch.exp(-(BASE_HEIGHT - height).abs() / .02),
            'lin_vel_z': lin[:, 2].square(),
            'ang_vel_xy': ang[:, :2].square().sum(-1),
            'torques': effort.square().sum(-1),
            'hip_torque_max': torque_max['hip'], 'thigh_torque_max': torque_max['thigh'],
            'calf_torque_max': torque_max['calf'],
            'joint_acc': acceleration.square().sum(-1),
            'action_rate': (obs.executor.output - obs.executor.previous_output).square().sum(-1),
            'joint_power': (velocity * effort).abs().sum(-1),
            'feet_slide': (foot_lateral * contact).sum(-1) * tilt,
            'hip_deviation': offset[:, index['hip']].abs().sum(-1) * (upright > .7),
            'leg_deviation': offset[:, torch.cat((index['thigh'], index['calf']))].abs().sum(-1) * (upright > .7),
            'arm_deviation': (arm_position - arm_default).abs().sum(-1) * (upright > .7),
            'joint_mirror': mirror * tilt,
            'feet_air_time': ((last_air - AIR_TIME_THRESHOLD) * first_contact).sum(-1) * moving,
            'feet_long_air': (air - MAX_AIR_TIME).clamp(min=0).square().sum(-1),
            'air_time_variance': last_air.clamp(max=.5).var(-1) + last_contact.clamp(max=.5).var(-1),
            'flat_orientation': gravity[:, :2].square().sum(-1),
        }
        reward = sum(WEIGHTS[name] * value for name, value in terms.items())
        base_contact = obs.base_contact.contact_force_samples.norm(dim=-1).amax((1, 2)) > .5
        bad_orientation = torch.acos(upright.clamp(-1., 1.)) > 1.
        metrics = {'ee_position_error_m': ee_error.sqrt(), 'ee_orientation_error_rad': ee_angle,
                   'lin_vel_xy_error_m_s': (command[:, :2] - lin[:, :2]).norm(dim=-1),
                   'ang_vel_z_error_rad_s': (command[:, 2] - ang[:, 2]).abs(),
                   'base_height_m': height,
                   **{'reward_' + name: WEIGHTS[name] * value for name, value in terms.items()}}
        return {'reward': reward, 'failure': base_contact | bad_orientation, 'metrics': metrics}
