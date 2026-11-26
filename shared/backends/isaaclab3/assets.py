"""Isaac Lab 3 native assets for the shared factory names.

Spawners use the solver-common UsdPhysics schemas so Newton reads the same scene that PhysX would;
PhysX-only property classes are left to the presets that already carry both variants.
"""

from functools import cache
from pathlib import Path
from typing import Literal

from shared.assembly.assets import (FACTORY_SCOPES, GO2_SUPERMIMIC, CameraData, Go2Actuators,  # noqa: F401
                                   ball_parameters, box_parameters, camera_parameters, checked, checked_gains,
                                   go2_parameters, workspace_file)
from shared.asset_formats import nucleus_copy
# The robots kept in this workspace (robots/) and the props below are written in Isaac Lab's
# RigidBodyPropertiesCfg / ArticulationCfg form, which Isaac Lab 3 keeps through its PhysX-schema
# aliases; none carries a quaternion, and their root poses are left to scene.py (xyzw). PhysX-only
# fields (solver iterations, combine modes) apply on OvPhysX and are ignored by Newton, as in Isaac
# Lab 3's own presets.

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


def franka_panda_usd():
    return nucleus_copy("franka_panda", "Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd")


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


@checked(go2_parameters)
def go2(actuators: Go2Actuators = "native"):
    """Isaac Lab 3's UNITREE_GO2_CFG on the vendored Isaac 5.1 USD shared with the isaac backend.

    Its DC motor preset differs from Isaac Lab 2.x (calf effort limit 45.43 N m), so behaviour and
    trained weights are not interchangeable with the isaac backend's go2. ``actuators='supermimic'``
    swaps in GO2_SUPERMIMIC's delayed DC motors and stance; Newton cannot author those, so the scene
    then runs Isaac Lab's explicit actuator models (environment.py).
    """
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    go2_parameters(actuators)
    config = UNITREE_GO2_CFG.copy()
    config.spawn.usd_path = str(nucleus_copy("go2", "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd"))
    if actuators == "supermimic":
        from .actuators import DelayedDCMotorCfg
        low, high = GO2_SUPERMIMIC["delay"]
        config.init_state.joint_pos = dict(GO2_SUPERMIMIC["stance"])
        config.actuators = {
            name: DelayedDCMotorCfg(joint_names_expr=list(group["joints"]), stiffness=group["stiffness"],
                                    damping=group["damping"], effort_limit=group["effort_limit"],
                                    saturation_effort=group["saturation_effort"],
                                    velocity_limit=group["velocity_limit"], armature=group["armature"],
                                    friction=0., min_delay=low, max_delay=high)
            for name, group in GO2_SUPERMIMIC["groups"].items()}
    return config


def anymal_c():
    """Isaac Lab 3's ANYMAL_C_CFG (ANYdrive LSTM legs) on the vendored Isaac 5.1 USD and actuator network."""
    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG

    config = ANYMAL_C_CFG.copy()
    config.spawn.usd_path = str(nucleus_copy("anymal_c", "Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd"))
    config.actuators["legs"].network_file = str(
        nucleus_copy("anymal_c", "Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"))
    return config


FrankaTracking = Literal['native', 'stiff']


def franka_parameters(fix_root_link=True, tracking='native'):
    if tracking not in FrankaTracking.__args__:
        raise ValueError(f'franka_panda.tracking must be one of {list(FrankaTracking.__args__)}')


@checked(franka_parameters)
def franka_panda(fix_root_link=True, tracking: FrankaTracking = 'native'):
    """Isaac Lab 3's FRANKA_PANDA_LEGACY_CFG, whose USD is the vendored panda_instanceable.usd.

    ``tracking: stiff`` selects its FRANKA_PANDA_LEGACY_HIGH_PD_CFG instead (arm stiffness 400, damping 80,
    gravity off on the robot links), the preset Isaac Lab's differential-IK tasks use: increments on the
    measured joint positions then do not let the arm sag. Scene building keeps gravity off on Newton too
    (scene.compensate_gravity)."""
    from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_CFG, FRANKA_PANDA_LEGACY_HIGH_PD_CFG

    franka_parameters(fix_root_link, tracking)
    config = (FRANKA_PANDA_LEGACY_HIGH_PD_CFG if tracking == 'stiff' else FRANKA_PANDA_LEGACY_CFG).copy()
    config.spawn.usd_path = str(franka_panda_usd())
    # Lab 3's per-solver articulation fragments carry no fix_root_link; the spawner owns that topology flag.
    config.spawn.fix_root_link = fix_root_link
    return config


def _pd_actuators(usd_path, actuators):
    """Implicit PD actuator groups of usd_object(actuators=...), one per joint-name pattern."""
    from isaaclab.actuators import ImplicitActuatorCfg
    groups = {}
    for index, (pattern, gains) in enumerate((actuators or {}).items()):
        if not isinstance(gains, dict) or 'stiffness' not in gains or set(gains) - {'stiffness', 'damping', 'effort_limit'}:
            raise ValueError(f'{usd_path}: actuators[{pattern!r}] takes stiffness, damping and an optional effort_limit')
        limit = gains.get('effort_limit')
        groups[f'drive_{index}'] = ImplicitActuatorCfg(
            joint_names_expr=[pattern], stiffness=float(gains['stiffness']), damping=float(gains.get('damping', 0.0)),
            joint_effort_limit=None if limit is None else float(limit))
    return groups


def usd_object(usd_path, mass=None, scale=1.0, fix_root_link=False, springs: dict | None = None,
               exclude: list | None = None, prim: str | None = None, moved: dict | None = None,
               actuators: dict | None = None, couplings: dict | None = None):
    """Imported USD object, simulated as its UsdPhysics schemas say (``usd_body_structure``).

    One rigid body is a dynamic rigid object; jointed bodies are an articulation without actuators,
    whose joints move only under contact and the drives, limits and friction authored in the USD.
    ``springs`` (joint name -> ``{stiffness, damping, target}``) sets those joints' USD drive springs in a
    cached layer over the file (``joint_spring_layer``), e.g. for buttons: SimReady authors 0.1-10 N/m on
    about 1 mm of travel, far below the weight of the cap, so its buttons rest pressed in. Newton's
    MuJoCo-Warp solver integrates a stiff spring on a light part explicitly: keep stiffness / moving mass
    below about 1 / substep^2 (4e4 at 5 ms), else the joint diverges.
    ``couplings`` (follower joint -> ``{leader, ratio, offset}``, follower = ratio * leader + offset in rad or m)
    links joints physically, on Newton and OvPhysX alike (``joint_coupling_layer``): a trash can's lid opens as
    its pedal is pressed, a dial's needle turns with its knob.
    ``actuators`` drives the joints of an articulated object as a robot: joint-name pattern ->
    ``{stiffness, damping, effort_limit?}`` (N·m/rad and N·m·s/rad, prismatic N/m and N·s/m; the limit
    defaults to the USD's), Isaac Lab implicit PD actuators following the position targets (the executor's,
    or the default pose of a robot set to ``none``). Without it the joints follow only the USD's drives, so
    an imported robot whose USD authors no stiffness hangs limp.
    ``exclude`` lists prims (relative to the default prim) left out, e.g. objects removed from an indoor
    scene in the editor, and ``moved`` gives objects another pose, still static (object -> ``{position,
    orientation}`` in the entity frame; ``scene_edit_layer``, shared/assembly/composition.py).
    ``prim`` takes one object out of such a scene (relative to the default prim; ``object_layer``): the
    entity is that object alone, placed at its own origin (``object_pose``), static until it has a mass.
    ``fix_root_link`` fixes the object in place: the articulation root, or the rigid body made
    kinematic. A USD without rigid bodies stays static geometry (terrain tiles); with a ``mass`` its
    root becomes the rigid body. ``mass`` replaces the authored mass of a single body and is required
    when the USD authors none."""
    import isaaclab.sim as sim
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from shared.asset_formats import (authored_top_scale, joint_coupling_layer, joint_spring_layer, object_layer, scene_edit_layer,
                                      recorded_joint_order, usd_authored_mass, usd_body_structure, usd_joint_order,
                                      usd_root_offset, usd_zero_outside_limits)

    path = workspace_file(usd_path)
    structure = usd_body_structure(path)
    if prim is not None:
        if structure != 'static' or exclude or springs or moved or couplings:
            raise ValueError(f'{usd_path}: prim takes one object out of a static scene, as it is '
                             '(no exclude, moved or springs)')
        scale = tuple(scale) if isinstance(scale, (list, tuple)) else (float(scale),) * 3
        dynamic = mass is not None and not fix_root_link
        spawn = sim.UsdFileCfg(usd_path=str(object_layer(path, prim, convex=dynamic)), scale=scale)
        if mass is None and not fix_root_link:
            return AssetBaseCfg(prim_path="", spawn=spawn)
        if mass is None:
            raise ValueError(f'{usd_path}: a fixed or moving object taken out of a scene needs a mass')
        spawn.rigid_props = sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True if fix_root_link else None)
        spawn.mass_props = sim.MassPropertiesCfg(mass=mass)
        return RigidObjectCfg(prim_path="", spawn=spawn)
    offset = usd_root_offset(path) if structure != 'static' else None
    if offset is not None:
        # Isaac Lab writes the scene pose to the root body: the asset would start displaced by this offset.
        raise ValueError(f'{usd_path}: its root body is moved ({[round(v, 3) for v in offset]} m) or turned from the asset '
                         'frame; Isaac Lab would place that body, not the asset, at the scene pose. Wrap it with '
                         '`run.py asset simready` (which moves the body frame onto the asset origin)')
    scale = tuple(scale) if isinstance(scale, (list, tuple)) else (float(scale),) * 3
    # Isaac Lab replaces the top prim's scale with the spawn scale; keep the one the file authors
    # (e.g. the 0.01 unit conversion of a centimetre environment wrapper).
    scale = tuple(a * b for a, b in zip(scale, authored_top_scale(path)))
    if (springs or couplings) and structure != 'articulation':
        raise ValueError(f'{usd_path}: springs and couplings act on movable joints; this object has none')
    layer = scene_edit_layer(path, exclude, moved) if exclude or moved else path
    layer = joint_spring_layer(layer, springs) if springs else layer
    spawn = sim.UsdFileCfg(usd_path=str(joint_coupling_layer(layer, couplings) if couplings else layer), scale=scale)
    if structure == 'articulation':
        if mass is not None:
            raise ValueError(f'{usd_path}: an articulated object takes its link masses from the USD')
        spawn.articulation_props = sim.ArticulationRootPropertiesCfg(fix_root_link=fix_root_link)
        initial = ArticulationCfg.InitialStateCfg()
        joints = recorded_joint_order(path) or usd_joint_order(path)
        if not joints:
            # Parts held by fixed joints only: the default '.*' joint pattern would match nothing.
            initial.joint_pos, initial.joint_vel = {}, {}
        elif outside := usd_zero_outside_limits(path):
            # A joint that cannot be at 0 starts at its nearest limit (a scene's initial_state may set others).
            initial.joint_pos = {name: outside.get(name, 0.) for name in joints}
        return ArticulationCfg(prim_path="", spawn=spawn, init_state=initial, actuators=_pd_actuators(usd_path, actuators))
    if actuators:
        raise ValueError(f'{usd_path}: actuators drive movable joints; this object has none')
    if structure == 'static' and mass is None:
        return AssetBaseCfg(prim_path="", spawn=spawn)
    if mass is None and not fix_root_link and not usd_authored_mass(path):
        raise ValueError(f'{usd_path}: the USD authors no mass for its rigid body; set usd_object.mass')
    if structure == 'static' or fix_root_link:
        # A bare fragment creates the rigid body on the spawn prim when the asset has none.
        spawn.rigid_props = sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True if fix_root_link else None)
    if mass is not None:
        spawn.mass_props = sim.MassPropertiesCfg(mass=mass)
    return RigidObjectCfg(prim_path="", spawn=spawn)


def usd_articulation(usd_path, fix_root_link=True):
    """Kept for existing scenes: a jointed :func:`usd_object`, fixed by default."""
    from shared.asset_formats import usd_body_structure
    if usd_body_structure(workspace_file(usd_path)) != 'articulation':
        raise ValueError(f'{usd_path}: usd_articulation needs jointed bodies; use usd_object')
    return usd_object(usd_path, fix_root_link=fix_root_link)


GROUND_PLANE = "Isaac/IsaacLab/Environments/Grid/default_ground_plane_checker_v1/default_ground_plane.usda"
GROUND_PLANE_TILE = 2.0  # metres per texture repeat of the vendored grid, as in Isaac Lab 3


def _spawn_vendored_ground(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Isaac Lab's spawn_ground_plane plus the UV scaling it applies only to its remote default URL."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_ground_plane
    from pxr import Gf, UsdGeom

    prim = spawn_ground_plane(prim_path, cfg, translation, orientation, **kwargs)
    mesh = UsdGeom.Mesh(prim.GetStage().GetPrimAtPath(f"{prim_path}/Environment/Geometry"))
    u, v = (size / (2.0 * GROUND_PLANE_TILE) for size in cfg.size)
    UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Set(
        [Gf.Vec2f(-u, -v), Gf.Vec2f(u, -v), Gf.Vec2f(u, v), Gf.Vec2f(-u, v)])
    return prim


def ground_plane():
    """Isaac Lab 3's default checker ground plane from the vendored Isaac 6.2 copy, so runs need no download."""
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim.GroundPlaneCfg(
        usd_path=str(nucleus_copy("ground_plane", GROUND_PLANE)), func=_spawn_vendored_ground))


def dome_light(color=(0.75, 0.75, 0.75), intensity=3000.0):
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg

    return AssetBaseCfg(prim_path="/World/light", spawn=sim.DomeLightCfg(color=tuple(color), intensity=intensity))


def _material(friction):
    from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
    return RigidBodyMaterialBaseCfg(static_friction=friction, dynamic_friction=friction)


@checked(box_parameters)
def box(size: tuple[float, float, float] = (1., 1., 1.), mass: float | None = None,
        color: tuple[float, float, float] = (.5, .5, .5), friction: float = 1.):
    box_parameters(size, mass, color, friction)
    import isaaclab.sim as sim
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    spawn = sim.CuboidCfg(size=tuple(size), collision_props=sim.UsdPhysicsCollisionCfg(),
                          visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(color)),
                          physics_material=_material(friction))
    if mass is None:
        return AssetBaseCfg(prim_path='', spawn=spawn)
    spawn.rigid_props = sim.UsdPhysicsRigidBodyCfg()
    spawn.mass_props = sim.MassCfg(mass=mass)
    return RigidObjectCfg(prim_path='', spawn=spawn)


@checked(ball_parameters)
def ball(radius: float = .12, mass: float = .2,
         color: tuple[float, float, float] = (.2, .8, .3), friction: float = 1.):
    """Free sphere with a rigid root; radius and mass are explicit (SI)."""
    ball_parameters(radius, mass, color, friction)
    import isaaclab.sim as sim
    from isaaclab.assets import RigidObjectCfg
    spawn = sim.SphereCfg(radius=radius, collision_props=sim.UsdPhysicsCollisionCfg(),
                          visual_material=sim.PreviewSurfaceCfg(diffuse_color=tuple(color)),
                          physics_material=_material(friction),
                          rigid_props=sim.UsdPhysicsRigidBodyCfg(), mass_props=sim.MassCfg(mass=mass))
    return RigidObjectCfg(prim_path='', spawn=spawn)


def contact_sensor(update_period: float = 0.0, history_length: int = 0, track_air_time: bool = False,
                   force_threshold: float | None = None, **parameters):
    """Isaac Lab's ContactSensor on the mount body (or every body its pattern matches): net contact force
    per body, the last ``history_length`` steps, and air/contact time when ``track_air_time``."""
    from isaaclab.sensors import ContactSensorCfg
    if force_threshold is not None:  # else Isaac Lab's own default
        parameters["force_threshold"] = force_threshold
    return ContactSensorCfg(prim_path="", update_period=update_period, history_length=history_length,
                            track_air_time=track_air_time, **parameters)


def height_scanner(pattern: dict, update_period: float = 0.0, max_distance: float = 1e6,
                   ray_alignment: Literal['base', 'yaw', 'world'] = 'base', **parameters):
    """Isaac Lab 3's RayCaster (on Newton its shape-BVH NewtonRaycastSensor) with a grid pattern.

    The scene builder sets the mount body, the sensor pose as offset and ``global_world_only``
    from ``meshes``; Newton ignores ``mesh_prim_paths`` and hits shapes by world instead."""
    from isaaclab.sensors import RayCasterCfg, patterns
    for name in ('offset', 'mesh_prim_paths', 'global_world_only'):
        if name in parameters:
            raise ValueError(f'height_scanner.{name} is set from the entity pose and meshes')
    grid = dict(pattern)
    grid['size'] = tuple(grid['size'])
    # The sensor tracks its mount body prim; Lab 3 deprecates spawning a sensor-frame child.
    return RayCasterCfg(prim_path='', spawn=None, update_period=update_period, max_distance=max_distance,
                        ray_alignment=ray_alignment, **parameters, pattern_cfg=patterns.GridPatternCfg(**grid))


@checked(camera_parameters)
def camera(width: int = 160, height: int = 120, data_types: tuple[CameraData, ...] = ('rgb',),
           focal_length: float = 24.0, clipping_range: tuple[float, float] = (0.02, 50.0),
           depth_clipping: Literal['max', 'zero', 'none'] = 'max'):
    """Pinhole camera rendered by Isaac Lab 3's kitless Newton Warp renderer, on Newton and OvPhysX.

    The scene builder spawns it under its mount body, where Lab 3's frame view follows the body.
    Newton Warp writes depth 0 where a ray hits nothing within the far plane; 'max' turns that into
    the far value natively, 'none' is turned into inf by the reader as on Isaac. The near plane is
    not applied, so geometry closer than it stays visible."""
    camera_parameters(width, height, data_types, focal_length, clipping_range, depth_clipping)
    import isaaclab.sim as sim
    from isaaclab.sensors import CameraCfg
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    # Isaac rasterizes with back-face culling, so a camera inside its own housing sees out.
    renderer = NewtonWarpRendererCfg(depth_clipping_behavior=depth_clipping, enable_backface_culling=True)
    return CameraCfg(prim_path='', width=width, height=height, data_types=list(data_types),
                     update_latest_camera_pose=True, renderer_cfg=renderer,
                     offset=CameraCfg.OffsetCfg(convention='opengl'),
                     spawn=sim.PinholeCameraCfg(focal_length=focal_length, horizontal_aperture=20.955,
                                                clipping_range=tuple(clipping_range)))



@cache
def _ovrtx_camera_renderer():
    from isaaclab.utils import configclass
    from isaaclab_ov.renderers import OVRTXRendererCfg

    @configclass
    class OvrtxCameraRendererCfg(OVRTXRendererCfg):
        depth_clipping_behavior: Literal['max', 'zero', 'none'] = 'none'
        """The camera factory's depth_clipping; OVRTX does not clip, so the reader applies it."""
    return OvrtxCameraRendererCfg


def use_renderer(scene, renderer):
    """Render the scene cameras with ``renderer``: ``newton`` keeps the factory's Newton Warp
    renderer, ``ovrtx`` swaps in Isaac Lab 3's kitless OVRTX renderer with the same depth clipping."""
    if renderer == 'newton':
        return
    from isaaclab.sensors import CameraCfg
    for config in vars(scene).values():
        if isinstance(config, CameraCfg):
            clipping = config.renderer_cfg.depth_clipping_behavior
            config.renderer_cfg = _ovrtx_camera_renderer()(depth_clipping_behavior=clipping)

def frame_transformer(target_body, target_position=(0.0, 0.0, 0.0),
                      target_orientation=(1.0, 0.0, 0.0, 0.0), target_name="tcp"):
    """Measure a body frame relative to a sensor's source body on the same asset.

    Parameters keep the shared wxyz convention; Isaac Lab 3's OffsetCfg takes xyzw."""
    from isaaclab.sensors import FrameTransformerCfg, OffsetCfg
    from shared.assembly.spec import Mount, Pose
    from .frames import xyzw

    if not target_body:
        raise ValueError("frame_transformer requires a target_body")
    Mount(".", target_body)
    pose = Pose(tuple(target_position), tuple(target_orientation))
    return FrameTransformerCfg(prim_path="", target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{SOURCE_ASSET}/" + target_body, name=target_name,
        offset=OffsetCfg(pos=pose.position, rot=xyzw(pose.orientation)),
    )])


FACTORIES = {factory.__name__: factory for factory in (
    go2, anymal_c, franka_panda, m20, piper, piper_x, ground_plane, dome_light, clear_sky, terrain,
    seattle_table, dex_cube, box, ball,
    usd_robot, usd_articulation, usd_object, height_scanner, contact_sensor, frame_transformer, camera)}
# Global entities without collision shapes; ray casts never hit them.
NO_SHAPES = ('dome_light', 'clear_sky')
# Sensors this backend binds to a mount body; height_scanner also casts at ``meshes``.
SENSORS = ('height_scanner', 'contact_sensor', 'frame_transformer', 'camera')
RAY_SENSORS = ('height_scanner',)
# Sensors that may also stand fixed in the world, without a mount: a camera, spawned in its environment.
FREE_SENSORS = ('camera',)
