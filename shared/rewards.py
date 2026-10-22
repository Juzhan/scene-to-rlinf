"""本平台自定义的奖励项。

这些函数和 Isaac Lab 内置奖励项用法完全一致，直接挂到 ``RewardTermCfg`` 即可，
参数里同样支持 ``SceneEntityCfg``（由 RewardManager 自动解析）。
"""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def lateral_velocity_error_l2(env, command_name):
    command = env.command_manager.get_command(command_name)
    return (env.scene["robot"].data.root_lin_vel_b[:, 1] - command[:, 1]).square()


def yaw_velocity_error_l2(env, command_name):
    command = env.command_manager.get_command(command_name)
    return (env.scene["robot"].data.root_ang_vel_b[:, 2] - command[:, 2]).square()


def route_error_l2(env):
    from .observations import route_state

    return route_state(env).square().sum(dim=-1)


def obstacle_keepout(
    env,
    asset_cfg: SceneEntityCfg,
    obstacles: tuple[tuple[float, float, float], ...],
    safe_margin: float = 0.05,
) -> torch.Tensor:
    """惩罚机器人连杆进入竖直圆柱障碍物的"安全区"。

    和"撞上之后才罚"（接触力惩罚）不同，这里按**侵入深度**在接触之前就开始惩罚，
    策略能较早学到保持间距，而不用先撞出经验。表达式：

        penalty = Σ_连杆 Σ_障碍物 max(0, (r_障碍物 + margin) - d_连杆到轴线)

    Args:
        env: 环境实例。
        asset_cfg: 机器人实体配置，用 ``body_names`` 指定要检查哪些连杆。
        obstacles: 障碍物列表 ``(x, y, radius)``，坐标是**环境局部系**
            （障碍物本身也是按环境原点摆放的，所以比较前要把连杆位置减掉环境原点）。
        safe_margin: 安全余量（米），连杆到轴线距离小于 ``radius + safe_margin`` 时开始罚。

    Returns:
        形状 ``(num_envs,)`` 的惩罚值（各连杆越界深度之和，完全在安全区外时为 0）。
    """
    asset = env.scene[asset_cfg.name]
    # 世界系 -> 环境局部系：每个并行环境原点不同，而障碍物是相对各自原点摆放的
    body_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2] - env.scene.env_origins[:, None, :2]

    penalty = torch.zeros(env.num_envs, device=body_xy.device, dtype=body_xy.dtype)
    for ox, oy, radius in obstacles:
        center = torch.tensor([ox, oy], device=body_xy.device, dtype=body_xy.dtype)
        dist = torch.norm(body_xy - center, dim=-1)  # (num_envs, num_bodies)
        penalty += torch.clamp(radius + safe_margin - dist, min=0.0).sum(dim=-1)
    return penalty


def pendulum_upright(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """倒立摆"竖直程度"奖励：杆的局部 +z 轴与世界 +z 的夹角余弦（1=完全竖直）。

    Args:
        env: 环境实例。
        asset_cfg: 机器人实体配置，``body_names`` 指向倒立摆的杆。

    Returns:
        形状 ``(num_envs,)``，取值 [-1, 1]。
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (N, B, 4)
    num_bodies = quat.shape[1]
    axis_local = torch.zeros(quat.shape[0] * num_bodies, 3, device=quat.device, dtype=quat.dtype)
    axis_local[:, 2] = 1.0
    axis_w = math_utils.quat_apply(quat.reshape(-1, 4), axis_local)
    cos_tilt = axis_w[:, 2].reshape(quat.shape[0], num_bodies).mean(dim=-1)
    return cos_tilt


def pendulum_tilt_penalty(env, asset_cfg: SceneEntityCfg, threshold: float = 0.95) -> torch.Tensor:
    """倒立摆倾角超过阈值后按超出量惩罚（防止"杆倒下去了还在走路"）。"""
    cos_tilt = pendulum_upright(env, asset_cfg)
    return torch.clamp(threshold - cos_tilt, min=0.0)


def pendulum_fallen(env, asset_cfg: SceneEntityCfg, cos_threshold: float = 0.5) -> torch.Tensor:
    """终止条件：杆轴与竖直方向夹角超过 arccos(cos_threshold) 就判定"倒了"。"""
    return pendulum_upright(env, asset_cfg) < cos_threshold


def joint_deviation_from_default(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """关节偏离默认姿态的 L2（用于让机械臂保持"举杆"姿态附近）。"""
    asset = env.scene[asset_cfg.name]
    return torch.linalg.norm(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids], dim=-1
    )
