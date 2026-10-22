"""Piper X with the official parallel gripper and D435 model."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import CameraCfg


PIPER_X_USD_PATH = (
    Path(__file__).resolve().parents[4]
    / "assets/piper_x/piper_x.usda"
)


def piper_x_camera_config():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(PIPER_X_USD_PATH))
    cameras = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)]
    if len(cameras) != 1:
        raise ValueError(f"Expected one official Piper X camera, found {[str(prim.GetPath()) for prim in cameras]}")
    camera_path = cameras[0].GetPath().MakeRelativePath(stage.GetDefaultPrim().GetPath())
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/PiperX/{camera_path}",
        spawn=None,
        width=1280,
        height=720,
        update_period=1 / 30,
        update_latest_camera_pose=True,
        data_types=["rgb", "distance_to_image_plane"],
    )


PIPER_X_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(PIPER_X_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0, "joint2": 1.4, "joint3": -1.2,
            "joint4": 0.0, "joint5": 1.2, "joint6": 0.0,
            "gripper_joint1": 0.05, "gripper_joint2": -0.05,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=100.0, velocity_limit_sim=5.0,
            stiffness=400.0, damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint[12]"],
            effort_limit_sim=10.0, velocity_limit_sim=3.0,
            stiffness=1000.0, damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
