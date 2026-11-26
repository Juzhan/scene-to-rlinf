"""M20 articulation parameters; USD resources are configurable independently of Python code."""

import os
from pathlib import Path
import isaaclab.sim as sim
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

M20_USD = os.environ.get("M20_USD", str(
    Path(__file__).resolve().parents[4] / "assets/m20/M20.usd"
))

DEEPROBOTICS_M20_CFG = ArticulationCfg(
    spawn=sim.UsdFileCfg(
        usd_path=M20_USD,
        activate_contact_sensors=True,
        rigid_props=sim.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0,
            angular_damping=0.0, max_linear_velocity=1000.0, max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={".*hipx_joint": 0.0, "f[l,r]_hipy_joint": -0.3, "h[l,r]_hipy_joint": 0.3,
                   "f[l,r]_knee_joint": 0.6, "h[l,r]_knee_joint": -0.6, ".*wheel_joint": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "joint": DelayedPDActuatorCfg(
            joint_names_expr=[".*hipx_joint", ".*hipy_joint", ".*knee_joint"],
            effort_limit=76.4, velocity_limit=22.4, stiffness=80.0, damping=2.0,
            friction=0.0, armature=0.0, min_delay=0, max_delay=1,
        ),
        "wheel": DelayedPDActuatorCfg(
            joint_names_expr=[".*_wheel_joint"], effort_limit=21.6, velocity_limit=79.3,
            stiffness=0.0, damping=0.6, friction=0.0, armature=0.00243216, min_delay=0, max_delay=1,
        ),
    },
)
