"""Reusable native assets; resource locations stay behind named factories."""

from pathlib import Path
from typing import Literal

from shared.assembly.assets import (CameraData, ball_parameters, box_parameters, camera_parameters,
                                   checked, checked_gains, workspace_file)
from shared.asset_formats import nucleus_copy

ASSETS = Path(__file__).resolve().parents[3] / "assets"


def _local_robot(config, usd_path, fix_root_link):
    config = config.copy()
    if usd_path is not None:
        config.spawn.usd_path = str(Path(usd_path).expanduser().resolve())
    if not Path(config.spawn.usd_path).is_file():
        raise FileNotFoundError(f"Robot USD asset is missing: {config.spawn.usd_path}")
    config.spawn.articulation_props.fix_root_link = fix_root_link
    return config


def piper(usd_path=None, fix_root_link=True):
    from .robots.piper import PIPER_CFG
    return _local_robot(PIPER_CFG, usd_path, fix_root_link)


def _override_gains(config, gains):
    """gains maps exact joint names to [stiffness, damping], replacing the native values of those joints."""
    import re

    joints = list(config.init_state.joint_pos)
    unknown = set(gains) - set(joints)
    if unknown:
        raise ValueError(f"gains: unknown joints {sorted(unknown)}; expected names among {joints}")
    for actuator in config.actuators.values():
        names = [joint for joint in joints if any(re.fullmatch(expr, joint) for expr in actuator.joint_names_expr)]
        if set(names) & set(gains):
            actuator.stiffness = {re.escape(joint): float(gains.get(joint, (actuator.stiffness,))[0]) for joint in names}
            actuator.damping = {re.escape(joint): float(gains.get(joint, (0., actuator.damping))[1]) for joint in names}
    return config


@checked(lambda gains=None, **_: checked_gains(gains or {}))
def piper_x(usd_path=None, fix_root_link=True, gains=None):
    """gains: optional {joint: [stiffness, damping]} replacing PIPER_X_CFG's fixed-base arm gains."""
    from .robots.piper_x import PIPER_X_CFG
    config = _local_robot(PIPER_X_CFG, usd_path, fix_root_link)
    return _override_gains(config, checked_gains(gains)) if gains else config


def m20(usd_path=None, fix_root_link=False):
    from .robots.m20 import DEEPROBOTICS_M20_CFG
    return _local_robot(DEEPROBOTICS_M20_CFG, usd_path, fix_root_link)


def go2():
    """Native UNITREE_GO2_CFG on the vendored copy of its Isaac 5.1 USD (assets/go2/source)."""
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    config = UNITREE_GO2_CFG.copy()
    config.spawn.usd_path = str(nucleus_copy("go2", "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"))
    return config


def anymal_c():
    """Native ANYMAL_C_CFG on the vendored copies of its Isaac 5.1 USD and ANYdrive LSTM (assets/anymal_c/source)."""
    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

    config = _local_robot(ANYMAL_C_CFG, nucleus_copy("anymal_c", "Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd"),
                          False)
    config.actuators["legs"].network_file = str(
        nucleus_copy("anymal_c", "Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"))
    return config


def franka_panda(fix_root_link=True):
    """Native FRANKA_PANDA_CFG on the vendored copy of its Isaac 5.1 USD (assets/franka_panda/source)."""
    from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

    return _local_robot(FRANKA_PANDA_CFG, franka_panda_usd(), fix_root_link)


def franka_panda_usd():
    return nucleus_copy("franka_panda", "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd")


def ground_plane():
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim.GroundPlaneCfg())


def dome_light(color=(0.75, 0.75, 0.75), intensity=3000.0):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path="/World/light",
                        spawn=sim.DomeLightCfg(color=tuple(color), intensity=intensity))


def seattle_table():
    """Vendored copy of the Isaac 5.1 SeattleLabTable USD (assets/seattle_table/source)."""
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    usd = nucleus_copy("seattle_table", "Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    return AssetBaseCfg(prim_path="", spawn=sim.UsdFileCfg(usd_path=str(usd)))


def dex_cube(scale=(0.8, 0.8, 0.8)):
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg

    return RigidObjectCfg(prim_path="", spawn=sim.UsdFileCfg(
        usd_path=str(dex_cube_usd()),
        scale=tuple(scale),
        rigid_props=sim.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0, max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0, disable_gravity=False,
        ),
    ))


def dex_cube_usd():
    return nucleus_copy("dex_cube", "Isaac/Props/Blocks/DexCube/dex_cube_instanceable.usd")


def white_marble():
    import isaaclab.sim as sim

    return sim.MdlFileCfg(
        mdl_path=str(nucleus_copy("white_marble", "Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                                                  "TilesMarbleSpiderWhiteBrickBondHoned.mdl")),
        project_uvw=True, texture_scale=(0.25, 0.25),
    )


def clear_sky(intensity=750.0):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(
        prim_path="",
        spawn=sim.DomeLightCfg(
            intensity=intensity,
            texture_file=str(nucleus_copy("clear_sky", "Isaac/Materials/Textures/Skies/PolyHaven/"
                                                       "kloofendal_43d_clear_puresky_4k.hdr")),
        ),
    )


MATERIALS = {"white_marble": white_marble}


# Sub-terrain names used in scene.yaml -> Isaac Lab sub-terrain config classes.
# Mesh terrains build exact geometry; height-field terrains (rough, slope, waves,
# obstacles, stepping stones) are sampled on a grid and suit rough-ground training.
TERRAIN_VARIANTS = {
    "flat": "MeshPlaneTerrainCfg",
    "stairs": "MeshPyramidStairsTerrainCfg",
    "inverted_stairs": "MeshInvertedPyramidStairsTerrainCfg",
    "random_grid": "MeshRandomGridTerrainCfg",
    "rails": "MeshRailsTerrainCfg",
    "pit": "MeshPitTerrainCfg",
    "box": "MeshBoxTerrainCfg",
    "gap": "MeshGapTerrainCfg",
    "floating_ring": "MeshFloatingRingTerrainCfg",
    "star": "MeshStarTerrainCfg",
    "repeated_pyramids": "MeshRepeatedPyramidsTerrainCfg",
    "repeated_boxes": "MeshRepeatedBoxesTerrainCfg",
    "repeated_cylinders": "MeshRepeatedCylindersTerrainCfg",
    "rough": "HfRandomUniformTerrainCfg",
    "slope": "HfPyramidSlopedTerrainCfg",
    "inverted_slope": "HfInvertedPyramidSlopedTerrainCfg",
    "hf_stairs": "HfPyramidStairsTerrainCfg",
    "hf_inverted_stairs": "HfInvertedPyramidStairsTerrainCfg",
    "obstacles": "HfDiscreteObstaclesTerrainCfg",
    "waves": "HfWaveTerrainCfg",
    "stepping_stones": "HfSteppingStonesTerrainCfg",
}


def _as_tuples(value):
    if isinstance(value, (list, tuple)):
        return tuple(_as_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _as_tuples(item) for key, item in value.items()}
    return value


def terrain_variant(factory, parameters):
    if factory not in TERRAIN_VARIANTS:
        raise ValueError(f"Unknown terrain variant '{factory}'. Available: {', '.join(TERRAIN_VARIANTS)}")
    import isaaclab.terrains as terrains

    config_class = getattr(terrains, TERRAIN_VARIANTS[factory])
    values = _as_tuples(parameters)
    for key in ("object_params_start", "object_params_end"):
        if key in values:
            values[key] = config_class.ObjectCfg(**values[key])
    return config_class(**values)


def terrain(terrain_generator, physics_material, material, **parameters):
    import isaaclab.sim as sim
    import isaaclab.terrains as terrains

    generator = dict(terrain_generator)
    generator["size"] = tuple(generator["size"])
    generator["difficulty_range"] = tuple(generator["difficulty_range"])
    generator["sub_terrains"] = {
        name: terrain_variant(variant["factory"], variant.get("parameters", {}))
        for name, variant in generator["sub_terrains"].items()
    }
    return terrains.TerrainImporterCfg(
        prim_path="", **parameters,
        terrain_generator=terrains.TerrainGeneratorCfg(**generator),
        physics_material=sim.RigidBodyMaterialCfg(**physics_material),
        visual_material=MATERIALS[material](),
    )


def height_scanner(pattern, **parameters):
    from isaaclab.sensors import RayCasterCfg, patterns

    grid = dict(pattern)
    grid["size"] = tuple(grid["size"])
    return RayCasterCfg(prim_path="", **parameters, pattern_cfg=patterns.GridPatternCfg(**grid))


def contact_sensor(**parameters):
    from isaaclab.sensors import ContactSensorCfg

    return ContactSensorCfg(prim_path="", **parameters)


@checked(camera_parameters)
def camera(width: int = 160, height: int = 120, data_types: tuple[CameraData, ...] = ('rgb',),
           focal_length: float = 24.0, clipping_range: tuple[float, float] = (0.02, 50.0),
           depth_clipping: Literal['max', 'zero', 'none'] = 'max'):
    """Pinhole camera; every selected data type is rendered at width × height.

    Depth beyond the far clipping plane becomes the far value ('max'), 0 ('zero') or inf ('none').
    """
    camera_parameters(width, height, data_types, focal_length, clipping_range, depth_clipping)
    import isaaclab.sim as sim
    from isaaclab.sensors import CameraCfg

    return CameraCfg(prim_path="", width=width, height=height, data_types=list(data_types),
                     depth_clipping_behavior=depth_clipping, offset=CameraCfg.OffsetCfg(convention="opengl"),
                     spawn=sim.PinholeCameraCfg(focal_length=focal_length, horizontal_aperture=20.955,
                                                clipping_range=tuple(clipping_range)))


def frame_transformer(target_body, target_position=(0.0, 0.0, 0.0),
                      target_orientation=(1.0, 0.0, 0.0, 0.0), target_name="tcp"):
    """Measure a body frame relative to a sensor's source body on the same asset."""
    from isaaclab.sensors import FrameTransformerCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
    from shared.assembly.spec import Mount, Pose

    if not target_body:
        raise ValueError("frame_transformer requires a target_body")
    Mount(".", target_body)
    pose = Pose(tuple(target_position), tuple(target_orientation))
    return FrameTransformerCfg(prim_path="", target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{SOURCE_ASSET}/" + target_body, name=target_name,
        offset=OffsetCfg(pos=pose.position, rot=pose.orientation),
    )])


@checked(box_parameters)
def box(size: tuple[float, float, float] = (1., 1., 1.), mass: float | None = None,
        color: tuple[float, float, float] = (.5, .5, .5), friction: float = 1.):
    box_parameters(size, mass, color, friction)
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    spawn = sim.CuboidCfg(size=tuple(size), collision_props=sim.CollisionPropertiesCfg(),
                          visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(color)),
                          physics_material=sim.RigidBodyMaterialCfg(static_friction=friction,
                                                                   dynamic_friction=friction))
    if mass is None:
        return AssetBaseCfg(prim_path='', spawn=spawn)
    spawn.rigid_props = sim.RigidBodyPropertiesCfg()
    spawn.mass_props = sim.MassPropertiesCfg(mass=mass)
    return RigidObjectCfg(prim_path='', spawn=spawn)


@checked(ball_parameters)
def ball(radius: float = .12, mass: float = .2,
         color: tuple[float, float, float] = (.2, .8, .3), friction: float = 1.):
    """Free sphere with a rigid root; radius and mass are explicit (SI)."""
    ball_parameters(radius, mass, color, friction)
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    spawn = sim.SphereCfg(radius=radius, collision_props=sim.CollisionPropertiesCfg(),
                          visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(color)),
                          physics_material=sim.RigidBodyMaterialCfg(static_friction=friction,
                                                                   dynamic_friction=friction))
    spawn.rigid_props = sim.RigidBodyPropertiesCfg()
    spawn.mass_props = sim.MassPropertiesCfg(mass=mass)
    return RigidObjectCfg(prim_path='', spawn=spawn)


def urdf_robot(urdf_path, fix_base=True, stiffness=100.0, damping=10.0, merge_fixed_joints=True):
    """Imported URDF articulation; every joint shares one implicit PD drive, tune it per robot."""
    import hashlib
    import isaaclab.sim as sim
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    path = workspace_file(urdf_path)
    # The converted USD is a disposable cache next to the other editor caches.
    output = Path(__file__).resolve().parents[3] / ".cache/urdf_usd" / (
        path.stem + "_" + hashlib.sha1(str(path).encode()).hexdigest()[:8])
    gains = sim.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=stiffness, damping=damping)
    spawn = sim.UrdfFileCfg(asset_path=str(path), usd_dir=str(output), usd_file_name=path.stem + ".usd",
                            fix_base=fix_base, merge_fixed_joints=merge_fixed_joints,
                            joint_drive=sim.UrdfFileCfg.JointDriveCfg(gains=gains))
    return ArticulationCfg(prim_path="", spawn=spawn, init_state=ArticulationCfg.InitialStateCfg(),
                           actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                                    stiffness=stiffness, damping=damping)})


def usd_robot(usd_path, fix_root_link=True, stiffness=None, damping=None):
    """Imported USD articulation; joint drives come from the USD unless gains are given."""
    import isaaclab.sim as sim
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    spawn = sim.UsdFileCfg(usd_path=str(workspace_file(usd_path)),
                           articulation_props=sim.ArticulationRootPropertiesCfg(fix_root_link=fix_root_link))
    return ArticulationCfg(prim_path="", spawn=spawn, init_state=ArticulationCfg.InitialStateCfg(),
                           actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"],
                                                                    stiffness=stiffness, damping=damping)})


def usd_object(usd_path, mass=None, scale=1.0):
    """Imported USD prop: static without a mass; with a mass the USD must already carry rigid-body physics."""
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    scale = tuple(scale) if isinstance(scale, (list, tuple)) else (float(scale),) * 3
    spawn = sim.UsdFileCfg(usd_path=str(workspace_file(usd_path)), scale=scale)
    if mass is None:
        return AssetBaseCfg(prim_path="", spawn=spawn)
    spawn.mass_props = sim.MassPropertiesCfg(mass=mass)
    return RigidObjectCfg(prim_path="", spawn=spawn)


FACTORIES = {factory.__name__: factory for factory in
             (go2, anymal_c, terrain, height_scanner, contact_sensor, clear_sky,
              franka_panda, ground_plane, dome_light, seattle_table, dex_cube, camera, frame_transformer,
              piper, piper_x, m20, box, ball, urdf_robot, usd_robot, usd_object)}

# Native manager-scene defaults belong to named presets, not inferred SDK types.
# Explicit scene scope overrides these defaults; unlisted factories are per-env.
FACTORY_SCOPES = {name: "global" for name in
                  ("terrain", "ground_plane", "dome_light", "clear_sky")}
