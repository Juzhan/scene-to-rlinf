"""松灵（AgileX）Piper 六轴机械臂的 Isaac Lab 资产配置。

模型来源：官方 `agilexrobotics/piper_ros`（noetic 分支）的 `piper_description`，
原样存于 `assets/piper/source/`（保持仓库路径，SOURCE.json 记录提交与 sha256），
用 Isaac Lab 官方 URDF 导入器转成 USD：

    python <IsaacLab>/scripts/tools/convert_urdf.py \
        assets/piper/source/src/piper_description/urdf/piper_description.urdf \
        assets/piper/piper.usd \
        --merge-joints --fix-base --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none

转换时把驱动参数置零（驱动由本文件的 actuator 配置接管），--merge-joints 把
base_link / gripper_base 固定连接合并进相邻刚体。

关节结构（转 USD 后）：
    joint1..joint6  旋转关节（臂）
    joint7, joint8  平移关节（平行夹爪两指，行程各 0.035 m，对称运动）

单位：rad / m。以下姿态由 URDF 正运动学校验过，见 docs/04-piper-arm.md。
"""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_PROJECT_ROOT = str(Path(__file__).resolve().parents[4])
PIPER_USD_PATH = os.environ.get(
    "RP_PIPER_USD", os.path.join(_PROJECT_ROOT, "assets", "piper", "piper.usd")
)
"""Piper 的 USD 路径，可用环境变量 RP_PIPER_USD 覆盖。"""

# 末端 TCP（两指中点）在 link6 局部坐标系下的偏移，由 URDF 正运动学算出。
PIPER_TCP_OFFSET = (0.0, 0.0, 0.1358)
"""夹爪两指中点在 link6 坐标系中的位置，用于 FrameTransformer。"""

# 抓取预备姿态：TCP 位于 (0.30, 0, 0.18) m，夹爪接近方向大致竖直向下。
PIPER_READY_JOINT_POS = {
    "joint1": 0.0,
    "joint2": 1.4,
    "joint3": -1.2,
    "joint4": 0.0,
    "joint5": 1.2,
    "joint6": 0.0,
    "joint7": 0.035,
    "joint8": -0.035,
}

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=PIPER_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos=PIPER_READY_JOINT_POS,
    ),
    actuators={
        # 参数取自 URDF：臂关节 effort 100 N·m / velocity 5 rad/s
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=100.0,
            damping=5.0,
        ),
        # 夹爪：行程 0.035 m，effort 10 N（URDF 值）；刚度给大一点以便夹持
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[78]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=1.0e3,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Piper 机械臂（含平行夹爪）的配置。"""

