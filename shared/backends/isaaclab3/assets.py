"""Isaac Lab 3 native assets for the shared factory names.

Spawners use the solver-common UsdPhysics schemas so Newton reads the same scene that PhysX would;
PhysX-only property classes are left to the presets that already carry both variants.
"""

from functools import cache
from typing import Literal

from shared.assembly.assets import (GO2_SUPERMIMIC, CameraData, Go2Actuators, ball_parameters, box_parameters,
                                   camera_parameters, checked, go2_parameters)
from shared.asset_formats import nucleus_copy
# Scopes follow the isaac backend; its scene_metadata resolves native keys and prim paths for both.
from shared.backends.isaac.assets import FACTORY_SCOPES  # noqa: F401
# Isaac Lab 3 keeps these Isaac Lab 2 configurations through its deprecated PhysX-schema aliases;
# none carries a quaternion, and their root poses are left to scene.py (xyzw). PhysX-only schema
# fields (solver iterations, combine modes) apply on OvPhysX and are ignored by Newton, as in
# Isaac Lab 3's own presets.
from shared.backends.isaac.assets import (clear_sky, contact_sensor, dex_cube, m20, piper, piper_x,
                                          seattle_table, terrain, usd_object, usd_robot)


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


def franka_panda(fix_root_link=True):
    """Isaac Lab 3's FRANKA_PANDA_LEGACY_CFG, whose USD is the vendored panda_instanceable.usd."""
    from isaaclab_assets.robots.franka import FRANKA_PANDA_LEGACY_CFG
    from shared.backends.isaac.assets import franka_panda_usd

    config = FRANKA_PANDA_LEGACY_CFG.copy()
    config.spawn.usd_path = str(franka_panda_usd())
    # Lab 3's per-solver articulation fragments carry no fix_root_link; the spawner owns that topology flag.
    config.spawn.fix_root_link = fix_root_link
    return config


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


def height_scanner(pattern, **parameters):
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
    return RayCasterCfg(prim_path='', spawn=None, **parameters, pattern_cfg=patterns.GridPatternCfg(**grid))


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
    usd_robot, usd_object, height_scanner, contact_sensor, frame_transformer, camera)}
# Global entities without collision shapes; ray casts never hit them.
NO_SHAPES = ('dome_light', 'clear_sky')
# Sensors this backend binds to a mount body; height_scanner also casts at ``meshes``.
SENSORS = ('height_scanner', 'contact_sensor', 'frame_transformer', 'camera')
RAY_SENSORS = ('height_scanner',)
# Sensors that may also stand fixed in the world, without a mount: none, every sensor follows a body.
FREE_SENSORS = ()
