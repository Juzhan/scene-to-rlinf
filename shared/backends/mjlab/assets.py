"""MuJoCo counterparts of public asset names; no SDK imports at discovery time."""
from typing import Literal
from shared.assembly.assets import (GO2_SUPERMIMIC, CameraData, Go2Actuators, ball_parameters, box_parameters,
                                   camera_parameters, checked, checked_gains, go2_parameters, workspace_file)

# Factories that return mjlab sensors; only these may be mounted on an entity body.
SENSORS = ('camera', 'frame_transformer', 'contact_sensor', 'height_scanner')
# Sensors that ray-cast against the entities listed in their scene ``meshes``.
RAY_SENSORS = ('height_scanner',)
# Sensors that may also stand fixed in the world, without a mount.
FREE_SENSORS = ('camera',)


@checked(go2_parameters)
def go2(actuators: Go2Actuators = 'native'):
    """Go2 from the MJCF recorded for its vendored Isaac 5.1 USD (assets/go2), driven like the Isaac
    UNITREE_GO2_CFG: explicit DC motors (kp 25, kd 0.5, 23.5 N·m, 30 rad/s) and its initial pose.
    Masses, foot and head bodies and colliders are the USD's; like Isaac's explicit actuators the
    converted joints have no passive damping, friction or armature. ``actuators='supermimic'`` uses
    GO2_SUPERMIMIC's DC motors (with their armature), stance and delay wrapper instead."""
    from mjlab.actuator import DcMotorActuatorCfg, DelayedActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from shared.asset_formats import mjcf_for, nucleus_copy
    go2_parameters(actuators)
    usd = nucleus_copy('go2', 'Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd')
    if actuators == 'supermimic':
        low, high = GO2_SUPERMIMIC['delay']
        cfg = _mjcf_entity(mjcf_for(usd), False, joint_pos=dict(GO2_SUPERMIMIC['stance']))
        cfg.articulation = EntityArticulationInfoCfg(actuators=tuple(
            DelayedActuatorCfg(base_cfg=DcMotorActuatorCfg(
                target_names_expr=group['joints'], stiffness=group['stiffness'], damping=group['damping'],
                effort_limit=group['effort_limit'], saturation_effort=group['saturation_effort'],
                velocity_limit=group['velocity_limit'], armature=group['armature']),
                delay_min_lag=low, delay_max_lag=high)
            for group in GO2_SUPERMIMIC['groups'].values()))
    else:
        cfg = _mjcf_entity(mjcf_for(usd), False, joint_pos={
            '.*L_hip_joint': .1, '.*R_hip_joint': -.1, 'F[L,R]_thigh_joint': .8, 'R[L,R]_thigh_joint': 1.,
            '.*_calf_joint': -1.5})
        cfg.articulation = EntityArticulationInfoCfg(actuators=(DcMotorActuatorCfg(
            target_names_expr=('.*_hip_joint', '.*_thigh_joint', '.*_calf_joint'), stiffness=25., damping=.5,
            effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.),))
    cfg.init_state.pos = (0., 0., .4)
    return cfg


def ground_plane():
    import mujoco
    from mjlab.entity import EntityCfg
    def spec_fn():
        spec = mujoco.MjSpec()
        spec.worldbody.add_geom(name='ground', type=mujoco.mjtGeom.mjGEOM_PLANE,
                                size=[0, 0, .1], friction=[1., .005, .0001])
        return spec
    return EntityCfg(spec_fn=spec_fn)


@checked(box_parameters)
def box(size: tuple[float, float, float] = (1., 1., 1.), mass: float | None = None,
        color: tuple[float, float, float] = (.5, .5, .5), friction: float = 1.):
    box_parameters(size, mass, color, friction)
    import mujoco
    from mjlab.entity import EntityCfg
    def spec_fn():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name='body')
        if mass is not None:
            body.add_freejoint(name='root')
        attributes = {} if mass is None else {'mass': mass}
        body.add_geom(name='geometry', type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[v / 2 for v in size], rgba=[*color, 1.],
                      friction=[friction, .005, .0001], **attributes)
        return spec
    return EntityCfg(spec_fn=spec_fn)


@checked(ball_parameters)
def ball(radius: float = .12, mass: float = .2,
         color: tuple[float, float, float] = (.2, .8, .3), friction: float = 1.):
    """Free sphere with a free root joint; the radius and mass are explicit (SI)."""
    ball_parameters(radius, mass, color, friction)
    import mujoco
    from mjlab.entity import EntityCfg
    def spec_fn():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name='body')
        body.add_freejoint(name='root')
        body.add_geom(name='geometry', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[radius], mass=mass, rgba=[*color, 1.],
                      friction=[friction, .005, .0001])
        return spec
    return EntityCfg(spec_fn=spec_fn)


def _mjcf_entity(path, fixed, gains=None, joint_pos=None, mass=None, pads=None, contacts=None):
    """Entity from an MJCF converted by shared.asset_formats.

    gains maps exact joint names to (stiffness, damping, effort limit or None); joints
    without gains stay passive. A fixed entity is welded to the world, else it gets
    a free root joint. MuJoCo's parent filter does not apply to links of a welded root,
    so their contacts with it are excluded, as PhysX does for jointed links. A mass
    rescales the single body's inertial. pads maps exact body names to a massless
    (half size, position) box added to that body; contacts maps exact body names to
    MjsGeom attributes then set on every geom of that body."""
    from pathlib import Path
    import re
    import mujoco
    from mjlab.actuator import BuiltinPositionActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    path = Path(path)
    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(path))
        spec.assets = {item.relative_to(path.parent).as_posix(): item.read_bytes()
                       for item in (path.parent / 'assets').rglob('*') if item.is_file()}
        roots = list(spec.worldbody.bodies)
        if len(roots) != 1:
            raise ValueError(f'{path}: expected one root body, found {[body.name for body in roots]}')
        free = [joint for joint in roots[0].joints if joint.type == mujoco.mjtJoint.mjJNT_FREE]
        if fixed:
            for joint in free:
                spec.delete(joint)
            for child in roots[0].bodies:
                spec.add_exclude(bodyname1=roots[0].name, bodyname2=child.name)
        elif not free:
            roots[0].add_freejoint(name='root')
        for key in list(spec.keys):
            spec.delete(key)
        if mass is not None:
            bodies = [body for body in spec.bodies if body.name != 'world' and body.mass > 0]
            if len(bodies) != 1:
                raise ValueError(f'{path}: mass applies to a single rigid body, found {[b.name for b in bodies]}')
            bodies[0].inertia = [value * mass / bodies[0].mass for value in bodies[0].inertia]
            bodies[0].mass = mass
        for name, (size, position) in (pads or {}).items():
            body = spec.body(name)
            body.add_geom(name=f'{name}_pad', type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=position,
                          mass=0., rgba=list(body.geoms[0].rgba))
        for name, attributes in (contacts or {}).items():
            for geom in spec.body(name).geoms:
                for key, value in attributes.items():
                    setattr(geom, key, value)
        return spec
    spec_fn.asset_path = path  # robot.joint_layout finds the asset's joint_order from it
    groups = {}
    for joint, values in (gains or {}).items():
        groups.setdefault(values, []).append(re.escape(joint))
    actuators = tuple(BuiltinPositionActuatorCfg(target_names_expr=tuple(names), stiffness=stiffness,
                                                 damping=damping, effort_limit=effort)
                      for (stiffness, damping, effort), names in groups.items())
    return EntityCfg(spec_fn=spec_fn,
                     articulation=EntityArticulationInfoCfg(actuators=actuators) if actuators else None,
                     init_state=EntityCfg.InitialStateCfg(**({} if joint_pos is None else {'joint_pos': joint_pos})))


def _joint_names(path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    return [model.joint(index).name for index in range(model.njnt)
            if model.jnt_type[index] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]


@checked(lambda gains=None, **_: checked_gains(gains or {}))
def piper_x(usd_path=None, fix_root_link=True, gains=None):
    """Piper X from the MJCF recorded beside its USD; gains match the Isaac PIPER_X_CFG.

    gains optionally maps exact joint names to [stiffness, damping]. The default 400/40 arm
    gains suit a fixed base; on a base welded to a moving robot they excite a stiff
    oscillation of the light arm base, so mobile setups pass lower gains (on both backends).

    The fingers touch a held object only through the flat 2.5 x 1 cm tip face of their
    collision hulls. MuJoCo returns one contact point per mesh pair, which wanders over
    that face, so the object pivots and slips out where PhysX keeps a contact patch; a
    box pad flush with the face gives box-box multi-point contacts instead. The 25 g
    fingers squeeze with 10 N and under the default soft contact sink about 6 mm each
    into the object (grip opening 0.037 vs 0.048 in Isaac); stiffer finger contacts with
    priority (not mixed with the object's) keep the opening at 0.045-0.048."""
    from pathlib import Path
    from shared.asset_formats import mjcf_for
    usd = workspace_file(usd_path or Path(__file__).resolve().parents[3] / 'assets/piper_x/piper_x.usda')
    arm, gripper = (400., 40., 100.), (1000., 50., 10.)
    native = {**{f'joint{index}': arm for index in range(1, 7)}, 'gripper_joint1': gripper, 'gripper_joint2': gripper}
    gains = checked_gains(gains or {})
    if gains.keys() - native.keys():
        raise ValueError(f'gains: unknown joints {sorted(gains.keys() - native.keys())}; expected names among {list(native)}')
    return _mjcf_entity(mjcf_for(usd), fix_root_link,
                        {joint: (*gains.get(joint, values[:2]), values[2]) for joint, values in native.items()},
                        {'joint1': 0., 'joint2': 1.4, 'joint3': -1.2, 'joint4': 0., 'joint5': 0.,
                         'joint6': 0., 'gripper_joint1': .05, 'gripper_joint2': -.05},
                        pads={finger: ([.0115, .005, .0005], [0., -.005, .0005])
                              for finger in ('gripper_link1', 'gripper_link2')},
                        contacts={finger: {'solref': [.01, 1.], 'solimp': [.95, .99, .001, .5, 2.], 'priority': 1,
                                           'condim': 4, 'friction': [1., .02, .0001]}
                                  for finger in ('gripper_link1', 'gripper_link2')})


def piper(usd_path=None, fix_root_link=True):
    """Piper from the MJCF recorded beside its USD; gains and ready pose match the Isaac PIPER_CFG."""
    from pathlib import Path
    from shared.asset_formats import mjcf_for
    usd = workspace_file(usd_path or Path(__file__).resolve().parents[3] / 'assets/piper/piper.usd')
    arm, gripper = (100., 5., 100.), (1000., 50., 10.)
    return _mjcf_entity(mjcf_for(usd), fix_root_link,
                        {**{f'joint{index}': arm for index in range(1, 7)}, 'joint7': gripper, 'joint8': gripper},
                        {'joint1': 0., 'joint2': 1.4, 'joint3': -1.2, 'joint4': 0., 'joint5': 1.2,
                         'joint6': 0., 'joint7': .035, 'joint8': -.035})


def m20(usd_path=None, fix_root_link=False):
    """DEEP Robotics M20 from the MJCF recorded beside its USD, driven like DEEPROBOTICS_M20_CFG.

    Isaac's DelayedPDActuator is an explicit PD whose lag of 0-1 physics steps is drawn per
    environment at reset and held; here that is an ideal PD (motor) inside a delay wrapper
    that draws the lag on the first step after reset and never refreshes it in an episode."""
    from pathlib import Path
    from mjlab.actuator import DelayedActuatorCfg, IdealPdActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg
    from shared.asset_formats import mjcf_for
    usd = workspace_file(usd_path or Path(__file__).resolve().parents[3] / 'assets/m20/M20.usd')
    cfg = _mjcf_entity(mjcf_for(usd), fix_root_link, joint_pos={
        '.*hipx_joint': 0., 'f[lr]_hipy_joint': -.3, 'h[lr]_hipy_joint': .3,
        'f[lr]_knee_joint': .6, 'h[lr]_knee_joint': -.6, '.*_wheel_joint': 0.})
    delayed = lambda base: DelayedActuatorCfg(base_cfg=base, delay_target=('position', 'velocity', 'effort'),
                                              delay_min_lag=0, delay_max_lag=1,
                                              delay_update_period=2 ** 62, delay_per_env_phase=False)
    cfg.articulation = EntityArticulationInfoCfg(actuators=(
        delayed(IdealPdActuatorCfg(target_names_expr=('.*hipx_joint', '.*hipy_joint', '.*knee_joint'),
                                   stiffness=80., damping=2., effort_limit=76.4)),
        delayed(IdealPdActuatorCfg(target_names_expr=('.*_wheel_joint',), stiffness=0., damping=.6,
                                   effort_limit=21.6, armature=.00243216))))
    cfg.init_state.pos = (0., 0., .58)
    return cfg


def anymal_c():
    """ANYmal-C from the MJCF recorded for its vendored Isaac 5.1 USD, driven like ANYMAL_C_CFG
    by the same vendored ANYdrive LSTM network (assets/anymal_c/source)."""
    from mjlab.entity import EntityArticulationInfoCfg
    from shared.asset_formats import mjcf_for, nucleus_copy
    from .actuators import LstmActuatorCfg
    network = nucleus_copy('anymal_c', 'Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt')
    usd = nucleus_copy('anymal_c', 'Isaac/IsaacLab/Robots/ANYbotics/ANYmal-C/anymal_c.usd')
    cfg = _mjcf_entity(mjcf_for(usd), False, joint_pos={
        '.*HAA': 0., '.*F_HFE': .4, '.*H_HFE': -.4, '.*F_KFE': -.8, '.*H_KFE': .8})
    cfg.articulation = EntityArticulationInfoCfg(actuators=(LstmActuatorCfg(
        target_names_expr=('.*HAA', '.*HFE', '.*KFE'), network_file=str(network),
        saturation_effort=120., effort_limit=80., velocity_limit=7.5),))
    cfg.init_state.pos = (0., 0., .6)
    return cfg


def franka_panda(fix_root_link=True):
    """Franka Panda from the vendored Isaac 5.1 USD; gains and pose match FRANKA_PANDA_CFG."""
    from shared.asset_formats import mjcf_for, nucleus_copy
    usd = nucleus_copy('franka_panda', 'Isaac/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd')
    shoulder, forearm, hand = (80., 4., 87.), (80., 4., 12.), (2e3, 1e2, 200.)
    return _mjcf_entity(mjcf_for(usd), fix_root_link,
                        {**{f'panda_joint{index}': shoulder for index in range(1, 5)},
                         **{f'panda_joint{index}': forearm for index in range(5, 8)},
                         'panda_finger_joint1': hand, 'panda_finger_joint2': hand},
                        {'panda_joint1': 0., 'panda_joint2': -.569, 'panda_joint3': 0., 'panda_joint4': -2.81,
                         'panda_joint5': 0., 'panda_joint6': 3.037, 'panda_joint7': .741,
                         'panda_finger_joint1': .04, 'panda_finger_joint2': .04})


def seattle_table():
    """Static Seattle lab table from the vendored Isaac 5.1 USD (collider: its top box)."""
    from shared.asset_formats import mjcf_for, nucleus_copy
    return _mjcf_entity(mjcf_for(nucleus_copy('seattle_table', 'Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd')),
                        True)


def urdf_robot(urdf_path, fix_base=True, stiffness=100.0, damping=10.0, merge_fixed_joints=True):
    """MJCF compiled from the imported URDF; every joint shares one PD drive like the Isaac factory.

    merge_fixed_joints only renames/merges Isaac links: MuJoCo keeps fixed-joint links as
    bodies welded to their parent, which is physically the same rigid body."""
    from shared.asset_formats import mjcf_for
    path = mjcf_for(workspace_file(urdf_path))
    return _mjcf_entity(path, fix_base, {joint: (stiffness, damping, None) for joint in _joint_names(path)})


def usd_robot(usd_path, fix_root_link=True, stiffness=None, damping=None):
    """MJCF converted from the imported USD; drives come from the USD unless gains are given."""
    from shared.asset_formats import mjcf_for, usd_drives
    usd = workspace_file(usd_path)
    path = mjcf_for(usd)
    drives = usd_drives(usd)
    gains = {}
    for joint in _joint_names(path):
        drive = drives.get(joint, {'stiffness': 0., 'damping': 0., 'max_force': None})
        gains[joint] = (drive['stiffness'] if stiffness is None else stiffness,
                        drive['damping'] if damping is None else damping, drive['max_force'])
    return _mjcf_entity(path, fix_root_link, gains)


def usd_object(usd_path, mass=None, scale=1.0):
    """MJCF converted from the imported USD prop: static without a mass, free body with one."""
    from shared.asset_formats import mjcf_for
    if scale != 1.0 and scale != [1.0] * 3 and scale != (1.0,) * 3:
        raise NotImplementedError('mjlab usd_object does not scale converted meshes yet; use scale 1')
    return _mjcf_entity(mjcf_for(workspace_file(usd_path)), mass is None, mass=mass)


def dex_cube(scale=(0.8, 0.8, 0.8)):
    """Isaac 5.1 DexCube stand-in: 6 cm cube of 0.216 kg (the USD's authored mass), scaled per axis.

    Only size, mass and friction are matched; the textured faces become one flat colour."""
    import mujoco
    from mjlab.entity import EntityCfg
    if len(scale) != 3 or any(not v > 0 for v in scale):
        raise ValueError('dex_cube.scale requires three positive factors')
    def spec_fn():
        spec = mujoco.MjSpec()
        body = spec.worldbody.add_body(name='body')
        body.add_freejoint(name='root')
        body.add_geom(name='geometry', type=mujoco.mjtGeom.mjGEOM_BOX, size=[.03 * v for v in scale],
                      mass=.216, rgba=[.85, .35, .2, 1.], friction=[1., .005, .0001])
        return spec
    return EntityCfg(spec_fn=spec_fn)


def clear_sky(intensity=750.0):
    """Directional light in place of Isaac's sky dome; brightness scales with intensity / 750.

    The dome has no direction; a vertical light avoids the shadow acne an oblique one leaves on planes."""
    import mujoco
    from mjlab.entity import EntityCfg
    if not intensity >= 0:
        raise ValueError('clear_sky.intensity must be nonnegative')
    def spec_fn():
        spec = mujoco.MjSpec()
        level = .6 * intensity / 750.
        spec.worldbody.add_light(name='sky', type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 pos=[0, 0, 5], dir=[0, 0, -1], diffuse=[level] * 3,
                                 ambient=[.2 * level] * 3, castshadow=True)
        return spec
    return EntityCfg(spec_fn=spec_fn)


@checked(camera_parameters)
def camera(width: int = 160, height: int = 120, data_types: tuple[CameraData, ...] = ('rgb',),
           focal_length: float = 24.0, clipping_range: tuple[float, float] = (0.02, 50.0),
           depth_clipping: Literal['max', 'zero', 'none'] = 'max'):
    """Native mujoco_warp camera with the Isaac pinhole's field of view (20.955 mm horizontal aperture).

    The scene builder sets its name and pose; a mounted camera follows its body. Depth beyond
    the far plane (or hitting nothing) becomes far / 0 / inf as in Isaac; the near plane is not
    applied, so geometry closer than it stays visible."""
    camera_parameters(width, height, data_types, focal_length, clipping_range, depth_clipping)
    import math
    from dataclasses import dataclass
    from mjlab.sensor import CameraSensorCfg
    if 'normals' in data_types:
        raise NotImplementedError('mjlab cameras render rgb and depth only; normals are not available')
    @dataclass(kw_only=True)
    class ClippedCameraCfg(CameraSensorCfg):
        clipping_range: tuple[float, float]
        depth_clipping: str
    native = tuple(dict.fromkeys('rgb' if kind == 'rgb' else 'depth' for kind in data_types))
    return ClippedCameraCfg(name='', width=width, height=height, data_types=native,
                            fovy=math.degrees(2 * math.atan(20.955 * height / width / (2 * focal_length))),
                            clipping_range=tuple(clipping_range), depth_clipping=depth_clipping)


def frame_transformer(target_body, target_position=(0.0, 0.0, 0.0),
                      target_orientation=(1.0, 0.0, 0.0, 0.0), target_name="tcp"):
    """Native MuJoCo framepos sensor on a site at the offset from target_body of the mount entity."""
    from dataclasses import dataclass
    import mujoco
    from mjlab.sensor import BuiltinSensor, SensorCfg
    from shared.assembly.spec import Pose
    if not target_body or any(character in target_body for character in '/.*?[]()|+\\^$'):
        raise NotImplementedError('mjlab frame_transformer needs one plain MJCF body name as target_body')
    pose = Pose(tuple(target_position), tuple(target_orientation))
    class FrameSensor(BuiltinSensor):
        def __init__(self, cfg):
            super().__init__(name=cfg.name)
            self.frame = cfg
        def edit_spec(self, scene_spec, entities):
            body = scene_spec.body(f'{self.frame.entity}/{self.frame.target_body}')
            if body is None:
                raise ValueError(f'{self.frame.name}: {self.frame.entity} has no body {self.frame.target_body}')
            # Group 5 keeps the marker site out of the default viewer and camera images.
            body.add_site(name=self.frame.name, pos=pose.position, quat=pose.orientation, group=5)
            scene_spec.add_sensor(name=self.frame.name, type=mujoco.mjtSensor.mjSENS_FRAMEPOS,
                                  objtype=mujoco.mjtObj.mjOBJ_SITE, objname=self.frame.name)
    @dataclass(kw_only=True)
    class FrameSensorCfg(SensorCfg):
        target_body: str
        target_name: str
        entity: str = ''  # scene key of the mount entity, set by the scene builder
        def build(self):
            return FrameSensor(self)
    return FrameSensorCfg(name='', target_body=target_body, target_name=target_name)


def contact_sensor(update_period=0.0, history_length=0, track_air_time=False):
    """Native mjlab net contact force on each body of the mount entity matched by the mount body pattern.

    Forces are summed per body in world axes like Isaac's net_forces_w; the history keeps the last
    history_length physics steps, most recent first. Only per-step updates exist, so update_period
    must be 0 or the physics dt (checked by the scene builder)."""
    from dataclasses import dataclass
    from mjlab.sensor import ContactMatch, ContactSensorCfg
    if not isinstance(history_length, int) or history_length < 0:
        raise ValueError('contact_sensor.history_length must be a nonnegative integer')
    if not update_period >= 0:
        raise ValueError('contact_sensor.update_period must be nonnegative')
    @dataclass(kw_only=True)
    class BodyContactCfg(ContactSensorCfg):
        update_period: float
    return BodyContactCfg(name='', primary=ContactMatch(mode='body', pattern=''), fields=('found', 'force'),
                          reduce='netforce', history_length=history_length, track_air_time=bool(track_air_time),
                          update_period=float(update_period))


def height_scanner_parameters(pattern, update_period, history_length, debug_vis, ray_alignment,
                              attach_yaw_only, max_distance, drift_range, ray_cast_drift_range,
                              visualizer_cfg, offset, mesh_prim_paths):
    """SDK-free check of the Isaac RayCasterCfg fields the public height_scanner accepts."""
    from math import isfinite
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)
    if not isinstance(pattern, dict) or {'resolution', 'size'} - pattern.keys():
        raise ValueError('height_scanner.pattern needs resolution and size (Isaac GridPatternCfg)')
    if pattern.keys() - {'resolution', 'size', 'direction', 'ordering'}:
        raise NotImplementedError('mjlab height_scanner.pattern supports resolution, size, direction and ordering, '
                                  f'not {sorted(pattern.keys() - {"resolution", "size", "direction", "ordering"})}')
    if not finite(pattern['resolution']) or pattern['resolution'] <= 0:
        raise ValueError('height_scanner.pattern.resolution must be a positive number of metres')
    size = pattern['size']
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(finite(v) and v >= 0 for v in size):
        raise ValueError('height_scanner.pattern.size needs two nonnegative lengths (m)')
    direction = pattern.get('direction', (0., 0., -1.))
    if (not isinstance(direction, (list, tuple)) or len(direction) != 3 or not all(finite(v) for v in direction)
            or not any(direction)):
        raise ValueError('height_scanner.pattern.direction needs three finite values, not all zero')
    if pattern.get('ordering', 'xy') not in ('xy', 'yx'):
        raise ValueError("height_scanner.pattern.ordering must be 'xy' or 'yx'")
    if not finite(update_period) or update_period < 0:
        raise ValueError('height_scanner.update_period must be nonnegative (s)')
    if type(history_length) is not int or history_length < 0:
        raise ValueError('height_scanner.history_length must be a nonnegative integer')
    if history_length:
        raise NotImplementedError('height_scanner.history_length: the Isaac RayCaster keeps no history '
                                  '(the value only switches it to eager updates); mjlab has no counterpart, use 0')
    if not isinstance(debug_vis, bool):
        raise ValueError('height_scanner.debug_vis must be true or false')
    if ray_alignment not in ('base', 'yaw', 'world'):
        raise ValueError("height_scanner.ray_alignment must be 'base', 'yaw' or 'world'")
    if attach_yaw_only is not None and not isinstance(attach_yaw_only, bool):
        raise ValueError('height_scanner.attach_yaw_only must be true, false or absent')
    if not finite(max_distance) or max_distance <= 0:
        raise ValueError('height_scanner.max_distance must be positive')
    if (not isinstance(drift_range, (list, tuple)) or len(drift_range) != 2
            or not all(finite(v) for v in drift_range)):
        raise ValueError('height_scanner.drift_range needs two numbers')
    if ray_cast_drift_range is not None and (not isinstance(ray_cast_drift_range, dict) or not all(
            isinstance(bounds, (list, tuple)) and len(bounds) == 2 for bounds in ray_cast_drift_range.values())):
        raise ValueError('height_scanner.ray_cast_drift_range maps x/y/z to [low, high]')
    if any(drift_range) or any(any(bounds) for bounds in (ray_cast_drift_range or {}).values()):
        raise NotImplementedError('mjlab height_scanner has no sensor drift; drift_range and '
                                  'ray_cast_drift_range must stay zero')
    if visualizer_cfg is not None:
        raise NotImplementedError('height_scanner.visualizer_cfg is an Isaac marker config; '
                                  'mjlab draws its own rays when debug_vis is true')
    if offset is not None:
        raise ValueError('height_scanner.offset: set the sensor pose (relative to its mount body) instead')
    if mesh_prim_paths is not None:
        raise ValueError('height_scanner.mesh_prim_paths: list the target entity under the sensor meshes instead')


def isaac_grid(size, resolution, direction, ordering, position, orientation, device):
    """Ray starts and unit directions in the mount body frame, exactly as Isaac's RayCaster builds them.

    Same expressions as isaaclab grid_pattern (its arange end tolerance, so the ray count matches
    for any size/resolution, and its 'xy'/'yx' meshgrid order), then the sensor offset: the
    position moves the starts, the rotation turns only the directions (RayCaster._initialize_rays_impl).
    mjlab casts unit directions, so they are normalized; the scanner scales max_distance to match."""
    import torch
    indexing = ordering if ordering == 'xy' else 'ij'
    x = torch.arange(start=-size[0] / 2, end=size[0] / 2 + 1.0e-9, step=resolution, device=device)
    y = torch.arange(start=-size[1] / 2, end=size[1] / 2 + 1.0e-9, step=resolution, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing=indexing)
    starts = torch.zeros(grid_x.numel(), 3, device=device)
    starts[:, 0] = grid_x.flatten()
    starts[:, 1] = grid_y.flatten()
    starts += torch.tensor(list(position), device=device)
    quaternion = torch.tensor(list(orientation), device=device)  # wxyz
    vector = torch.tensor(list(direction), device=device)
    twice = torch.linalg.cross(quaternion[1:], vector) * 2  # isaaclab quat_apply
    rotated = vector + quaternion[0] * twice + torch.linalg.cross(quaternion[1:], twice)
    return starts, (rotated / rotated.norm()).expand(len(starts), 3).clone()


@checked(height_scanner_parameters)
def height_scanner(pattern, update_period=0.0, history_length=0, debug_vis=False, ray_alignment='base',
                   attach_yaw_only=None, max_distance=1e6, drift_range=(0.0, 0.0), ray_cast_drift_range=None,
                   visualizer_cfg=None, offset=None, mesh_prim_paths=None):
    """Native mjlab RayCastSensor with the Isaac RayCaster grid, offset, alignment and range.

    The scene builder attaches it to the mount body, applies the sensor pose as Isaac's offset
    and restricts hits to the single ``meshes`` entity. Isaac-only fields without an mjlab
    counterpart (drift, history, marker config) are rejected."""
    height_scanner_parameters(pattern, update_period, history_length, debug_vis, ray_alignment, attach_yaw_only,
                              max_distance, drift_range, ray_cast_drift_range, visualizer_cfg, offset, mesh_prim_paths)
    import math
    from dataclasses import dataclass
    from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
    if attach_yaw_only is not None:  # deprecated Isaac flag; overrides ray_alignment like RayCaster does
        ray_alignment = 'yaw' if attach_yaw_only else 'base'
    @dataclass(kw_only=True)
    class IsaacGridPatternCfg(GridPatternCfg):
        ordering: str = 'xy'
        position: tuple[float, float, float] = (0., 0., 0.)  # sensor offset in the mount body frame
        orientation: tuple[float, float, float, float] = (1., 0., 0., 0.)
        def generate_rays(self, mj_model, device):
            return isaac_grid(self.size, self.resolution, self.direction, self.ordering,
                              self.position, self.orientation, device)
    @dataclass(kw_only=True)
    class HeightScannerCfg(RayCastSensorCfg):
        update_period: float
        target: str = ''  # scene key of the ray target entity, set by the scene builder
    direction = tuple(float(v) for v in pattern.get('direction', (0., 0., -1.)))
    grid = IsaacGridPatternCfg(size=tuple(float(v) for v in pattern['size']), resolution=float(pattern['resolution']),
                               direction=direction, ordering=pattern.get('ordering', 'xy'))
    # Isaac's warp ray parameter is measured in direction lengths; mjlab's distance in metres.
    return HeightScannerCfg(name='', frame=ObjRef(type='body', name=''), pattern=grid, ray_alignment=ray_alignment,
                            max_distance=float(max_distance) * math.sqrt(sum(v * v for v in direction)),
                            debug_vis=debug_vis, update_period=float(update_period))


FACTORIES = {factory.__name__: factory for factory in
             (go2, anymal_c, ground_plane, box, ball, piper, piper_x, m20, franka_panda, seattle_table, urdf_robot, usd_robot,
              usd_object, dex_cube, clear_sky, camera, frame_transformer, contact_sensor, height_scanner)}
