"""Explicit actuator models Isaac Lab 3 does not ship; they run on Isaac Lab's Python actuator path,
which environment.py selects for every scene."""
import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass


class DelayedDCMotor(DCMotor):
    """DCMotor whose position/velocity/effort commands lag ``min_delay``-``max_delay`` physics steps,
    drawn per environment at reset, as DelayedPDActuator does for the ideal PD."""

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self._delay_buffers = tuple(DelayBuffer(cfg.max_delay, self._num_envs, device=self._device) for _ in range(3))

    def reset(self, env_ids):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        count = len(range(self._num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
        lags = torch.randint(self.cfg.min_delay, self.cfg.max_delay + 1, (count,), dtype=torch.int, device=self._device)
        for buffer in self._delay_buffers:
            buffer.set_time_lag(lags, env_ids)
            buffer.reset(env_ids)

    def compute(self, control_action, joint_pos, joint_vel):
        positions, velocities, efforts = self._delay_buffers
        control_action.joint_positions = positions.compute(control_action.joint_positions)
        control_action.joint_velocities = velocities.compute(control_action.joint_velocities)
        control_action.joint_efforts = efforts.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    class_type: type = DelayedDCMotor
    min_delay: int = 0
    max_delay: int = 0

