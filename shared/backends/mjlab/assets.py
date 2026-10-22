"""MuJoCo counterparts of public asset names; no SDK imports at discovery time."""
from typing import Literal
from shared.assembly.assets import CameraData, box_parameters, camera_parameters, checked, workspace_file

# Factories that return mjlab sensors; only these may be mounted on an entity body.
SENSORS = ('camera', 'frame_transformer')


def go2():
    from pathlib import Path
    import mujoco
    from mjlab.entity import EntityCfg, EntityArticulationInfoCfg
    from mjlab.actuator import XmlMotorActuatorCfg
    path = Path(__file__).resolve().parents[3] / 'assets/go2_mujoco/go2.xml'
    def spec_fn():
        spec = mujoco.MjSpec.from_file(str(path))
        spec.assets = {f'assets/{mesh.name}': mesh.read_bytes()
                       for mesh in (path.parent / 'assets').iterdir() if mesh.is_file()}
        # Authored initial state replaces Menagerie's home keyframe. Preserve
        # the original torque motors; control integration is a separate step.
        for key in list(spec.keys):
            spec.delete(key)
        return spec
    class MotorRobotCfg(EntityCfg):
        def build(self):
            entity = super().build()
            # mjlab 1.1.1 seeds actuator controls from joint positions. For
            # XML torque motors, initial effort must instead be zero.
            for key in entity.spec.keys:
                key.ctrl = [0.] * len(entity.spec.actuators)
            return entity
    return MotorRobotCfg(spec_fn=spec_fn,
        articulation=EntityArticulationInfoCfg(actuators=(XmlMotorActuatorCfg(target_names_expr=('.*_joint',)),)),
        init_state=EntityCfg.InitialStateCfg(
        pos=(0., 0., .42), joint_pos={'.*_hip_joint': 0.,
        '.*_thigh_joint': .8, '.*_calf_joint': -1.5}))


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


def _mjcf_entity(path, fixed, gains=None, joint_pos=None, mass=None):
    """Entity from an MJCF converted by shared.asset_formats.

    gains maps exact joint names to (stiffness, damping, effort limit or None); joints
    without gains stay passive. A fixed entity is welded to the world, else it gets
    a free root joint. A mass rescales the single body's inertial."""
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
        return spec
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


def piper_x(usd_path=None, fix_root_link=True):
    """Piper X from the MJCF recorded beside its USD; gains match the Isaac PIPER_X_CFG."""
    from pathlib import Path
    from shared.asset_formats import mjcf_for
    usd = workspace_file(usd_path or Path(__file__).resolve().parents[3] / 'assets/piper_x/piper_x.usda')
    arm, gripper = (400., 40., 100.), (1000., 50., 10.)
    return _mjcf_entity(mjcf_for(usd), fix_root_link,
                        {**{f'joint{index}': arm for index in range(1, 7)},
                         'gripper_joint1': gripper, 'gripper_joint2': gripper},
                        {'joint1': 0., 'joint2': 1.4, 'joint3': -1.2, 'joint4': 0., 'joint5': 1.2,
                         'joint6': 0., 'gripper_joint1': .05, 'gripper_joint2': -.05})


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


FACTORIES = {factory.__name__: factory for factory in
             (go2, ground_plane, box, piper_x, urdf_robot, usd_robot, usd_object,
              dex_cube, clear_sky, camera, frame_transformer)}
