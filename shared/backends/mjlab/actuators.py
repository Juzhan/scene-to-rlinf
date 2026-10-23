"""mjlab counterparts of Isaac actuator models that mjlab does not ship."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg


@dataclass(kw_only=True)
class LstmActuatorCfg(DcMotorActuatorCfg):
    """Isaac ActuatorNetLSTM: a TorchScript LSTM maps (target - position, velocity) to torque."""

    network_file: str
    stiffness: float = 0.
    damping: float = 0.

    def build(self, entity, target_ids, target_names):
        return LstmActuator(self, entity, target_ids, target_names)


class LstmActuator(DcMotorActuator[LstmActuatorCfg]):
    """Per env x joint hidden and cell states, zeroed for the environments that reset.

    Isaac's ActuatorNetLSTM never records the joint velocity its DC-motor clip reads, so
    the torque-speed curve stays at zero speed and torques are clipped to +-effort_limit;
    the velocity buffer here is left at zero for the same result."""

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self.network = torch.jit.load(self.cfg.network_file, map_location=device).eval()
        lstm = self.network.lstm.state_dict()
        shape = (len(lstm) // 4, data.nworld, len(self._target_names), lstm['weight_hh_l0'].shape[1])
        self._hidden = torch.zeros(shape, device=device)
        self._cell = torch.zeros(shape, device=device)

    def reset(self, env_ids=None):
        index = slice(None) if env_ids is None else env_ids
        self._hidden[:, index] = 0.
        self._cell[:, index] = 0.

    def compute(self, cmd):
        layers, envs, joints, width = self._hidden.shape
        sea_input = torch.stack((cmd.position_target - cmd.pos, cmd.vel), -1).reshape(envs * joints, 1, 2)
        with torch.inference_mode():
            torques, (hidden, cell) = self.network(
                sea_input, (self._hidden.reshape(layers, envs * joints, width),
                            self._cell.reshape(layers, envs * joints, width)))
        self._hidden.copy_(hidden.reshape(self._hidden.shape))
        self._cell.copy_(cell.reshape(self._cell.shape))
        return self._clip_effort(torques.reshape(envs, joints))
