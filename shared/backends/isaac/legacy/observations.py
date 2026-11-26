"""本平台自定义的观测项。"""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def route_state(env):
    robot = env.scene["robot"]
    lateral_position = robot.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    return torch.stack((lateral_position, robot.data.heading_w), dim=-1)


def pendulum_axis_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """倒立摆杆轴在**机体系**下的方向（3 维，竖直向上时为 (0,0,1)）。

    用机体系而不是世界系，策略学到的就是"相对机身怎么倾"，与朝向无关。
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, B, 4)
    num_bodies = quat.shape[1]
    axis_local = torch.zeros(quat.shape[0] * num_bodies, 3, device=quat.device, dtype=quat.dtype)
    axis_local[:, 2] = 1.0
    axis_w = math_utils.quat_apply(quat.reshape(-1, 4), axis_local).reshape(quat.shape[0], num_bodies, 3)
    axis_w = axis_w.mean(dim=1)  # 多根杆时取平均
    return math_utils.quat_apply_inverse(asset.data.root_quat_w, axis_w)


def goal_vector_b(env, command_name: str = "base_velocity") -> torch.Tensor:
    """目标点在机体系下的 xy 偏移 + 距离（3 维）。"""
    term = env.command_manager.get_term(command_name)
    delta_b, dist = term.goal_vector_b()
    return torch.cat((delta_b, dist.unsqueeze(-1)), dim=-1)
