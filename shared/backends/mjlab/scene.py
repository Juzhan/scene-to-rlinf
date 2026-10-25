"""Shared scene -> native mjlab SceneCfg. No policy or training loop."""
from math import isfinite
from shared.assembly.spec import Abstract, Pose

INITIAL_STATE = {'joint_positions', 'joint_velocities', 'linear_velocity', 'angular_velocity'}
# PhysX friction of a material against the default friction-1 material, per combine mode.
COMBINE = {'average': lambda value: (value + 1) / 2, 'min': lambda value: min(value, 1.),
           'multiply': lambda value: value, 'max': lambda value: max(value, 1.)}


def validate(description, settings):
    # Fail before starting the simulator rather than silently ignoring features.
    from .assets import SENSORS
    allowed = {'schema_version', 'name', 'entities', 'connections', 'simulation',
               'world', 'authoring', 'control', 'targets', 'application', 'observations', 'visualization'}
    extra = settings.keys() - allowed
    if extra:
        raise NotImplementedError(f'mjlab scene-only backend does not yet implement: {sorted(extra)}')
    sim = settings.get('simulation', {})
    # render_interval only paces Isaac's renderer; mjlab renders on request. `physx` holds PhysX
    # solver options and `mujoco` the native MujocoCfg options; each engine reads only its own.
    if sim.keys() - {'dt', 'gravity', 'render_interval', 'physics_material', 'physx', 'mujoco'}:
        raise NotImplementedError('mjlab scene simulation supports only dt, gravity, render_interval, '
                                  'physics_material and mujoco (physx is read by Isaac)')
    native = sim.get('mujoco', {})
    owned = {'timestep', 'gravity'}
    if not isinstance(native, dict) or native.keys() & owned:
        raise ValueError('simulation.mujoco holds MujocoCfg options; dt and gravity stay in simulation')
    # Option names are checked by MujocoCfg itself when the environment is built (static check stays SDK-free).
    material = sim.get('physics_material')
    # MuJoCo geoms carry their own friction (default 1) and contacts are inelastic,
    # which matches only an Isaac default material with friction 1 and no restitution.
    if material is not None and (material.get('static_friction', .5) != 1. or
            material.get('dynamic_friction', .5) != 1. or material.get('restitution', 0.) != 0.):
        raise NotImplementedError('mjlab physics_material supports only static/dynamic friction 1 and restitution 0')
    dt = sim.get('dt', .005)
    gravity = sim.get('gravity', [0., 0., -9.81])
    if not isinstance(dt, (int, float)) or not isfinite(dt) or dt <= 0:
        raise ValueError('simulation.dt must be positive and finite')
    if len(gravity) != 3 or not all(isfinite(x) for x in gravity):
        raise ValueError('simulation.gravity requires three finite values')
    world = settings.get('world', {})
    if world.keys() - {'env_spacing', 'contact_materials'}:
        raise NotImplementedError('mjlab scene world supports only env_spacing and contact_materials')
    for entity, values in (world.get('contact_materials') or {}).items():
        if values.get('restitution') != 0:
            raise NotImplementedError(f'world.contact_materials.{entity}: MuJoCo contacts are inelastic; '
                                      'mjlab supports only restitution 0')
    if (material or {}).get('friction_combine_mode', 'average') not in COMBINE:
        raise NotImplementedError(f'mjlab physics_material.friction_combine_mode must be one of {sorted(COMBINE)}')
    for path, (node, _) in description.index().items():
        if isinstance(node, Abstract):
            # Static and mounted abstract entities are read through shared readers.
            if node.reset is not None:
                raise NotImplementedError(f'{path}: mjlab randomized abstract entities (reset) are not connected yet; '
                                          'static and mounted abstract entities are readable')
            continue
        # prim_name only renames the Isaac prim; mjlab keys follow the logical path.
        # Robot body frames are read through the shared robot interface; nothing is compiled for them.
        if node.frames and node.kind != 'robot':
            raise NotImplementedError(f'{path}: mjlab body frames are read on robot entities only')
        if node.meshes:
            raise NotImplementedError(f'{path}: mjlab ray meshes are not implemented yet')
        if node.mount and node.factory not in SENSORS:
            raise NotImplementedError(f'{path}: mjlab mounts only sensors {list(SENSORS)}; physical attachment is not implemented')
        if node.reset is not None and node.factory in SENSORS:
            raise ValueError(f'{path}: sensors cannot be reset')
        if node.initial_state.keys() - INITIAL_STATE:
            raise NotImplementedError(f'{path}: mjlab initial_state supports only {sorted(INITIAL_STATE)}')


def build_scene_config(description, settings, factories):
    from dataclasses import replace
    import mujoco
    from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg, SensorCfg
    from mjlab.scene import SceneCfg
    from shared.backends.isaac.initial_state import apply_initial_state  # SDK-free value checks
    validate(description, settings)
    entities, sensors, keys = {}, {}, {}
    for path, (node, pose) in description.index().items():
        if not node.factory:
            continue
        key = path.removeprefix(description.root.name + '/').replace('/', '__')
        if key in keys.values():
            raise ValueError(f'Native scene key collision: {key}')
        cfg = factories[node.factory](**node.parameters)
        keys[path] = key
        if isinstance(cfg, SensorCfg):
            sensors[key] = (path, node, pose, cfg)
            continue
        if not node.pose_authored:
            pose = pose.compose(Pose(cfg.init_state.pos, cfg.init_state.rot))
        cfg.init_state.pos, cfg.init_state.rot = pose.position, pose.orientation
        if {'joint_positions', 'joint_velocities'} & node.initial_state.keys() and not cfg.spec_fn().joints:
            raise ValueError(f'{path}: this asset has no configurable joints')
        apply_initial_state(cfg, node.initial_state, path)
        if node.reset is not None and not any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in cfg.spec_fn().joints):
            raise ValueError(f'{path}: reset cannot move a fixed-base entity')
        entities[key] = cfg
    native_sensors = []
    for key, (path, node, pose, cfg) in sensors.items():
        mount = description.resolve_mount(path, node.mount) if node.mount else None
        if mount is not None and mount.entity not in keys:
            raise ValueError(f'{path}: mount entity {mount.entity} is not in the scene')
        # USD nests body prims; converted MJCF keeps each link name flat.
        body = f'{keys[mount.entity]}/{mount.body.rsplit("/", 1)[-1]}' if mount and mount.body else None
        if isinstance(cfg, CameraSensorCfg):
            # Isaac's opengl offset convention is MuJoCo's camera frame (-Z forward, +Y up);
            # a mounted pose is relative to its body, otherwise environment-local.
            local = node.pose if mount else pose
            if mount and (body is None or entities[keys[mount.entity]].spec_fn().body(body.split('/', 1)[1]) is None):
                raise ValueError(f'{path}: mount body {mount.body!r} is not a body of {mount.entity}')
            cfg = replace(cfg, name=key, parent_body=body, pos=local.position, quat=local.orientation)
        elif isinstance(cfg, ContactSensorCfg):
            if mount is None or not mount.body:
                raise ValueError(f'{path}: a contact sensor needs a mount entity and body pattern')
            dt = settings.get('simulation', {}).get('dt', .005)
            if cfg.update_period not in (0., dt):
                raise NotImplementedError(f'{path}: mjlab contact sensors update every physics step; '
                                          f'update_period must be 0 or {dt}')
            pattern = mount.body.rsplit('/', 1)[-1]
            if not entities[keys[mount.entity]].build().find_bodies(pattern)[0]:
                raise ValueError(f'{path}: mount body {mount.body!r} matches no body of {mount.entity}')
            cfg = replace(cfg, name=key, primary=ContactMatch(mode='body', pattern=pattern, entity=keys[mount.entity]))
        elif hasattr(cfg, 'target_body'):
            if mount is None:
                raise ValueError(f'{path}: frame targets are relative to a mount entity')
            cfg = replace(cfg, name=key, entity=keys[mount.entity])
        else:
            raise NotImplementedError(f'{path}: mjlab sensor {type(cfg).__name__} is not connected')
        native_sensors.append(cfg)
    sim = settings.get('simulation', {})
    mounted = [cfg.name for cfg in native_sensors if isinstance(cfg, CameraSensorCfg) and cfg.parent_body]
    welds = connection_welds(description, entities, keys, sim.get('dt', .005))
    materials = contact_friction(settings, keys)
    def configure(spec):
        spec.option.timestep = sim.get('dt', .005)
        spec.option.gravity = sim.get('gravity', [0., 0., -9.81])
        install_welds(spec, welds)
        apply_friction(spec, materials)
        if mounted:
            hide_camera_housings(spec, mounted)
    return SceneCfg(num_envs=1, env_spacing=settings.get('world', {}).get('env_spacing', 4.),
                    entities=entities, sensors=tuple(native_sensors), spec_fn=configure), keys


def root_body(cfg):
    """Name of the single root body of an entity spec and whether it has a free joint."""
    import mujoco
    spec = cfg.spec_fn()
    roots = list(spec.worldbody.bodies)
    if len(roots) != 1:
        raise ValueError(f'expected one root body, found {[body.name for body in roots]}')
    return roots[0].name, any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in roots[0].joints)


def connection_welds(description, entities, keys, dt):
    """Fixed connections as weld constraints; the child keeps its free root and is placed by a reset event.

    mjlab attaches every entity as its own tree and turns a fixed base into a mocap body, so a
    child cannot join its parent's kinematic tree. A weld with the stiffest stable time constant
    (2 dt) holds it instead; like a PhysX joint it disables contacts between the two bodies."""
    welds = []
    for joint in description.ordered_connections():
        for mount in (joint.child, joint.parent):
            if mount is not None and mount.entity not in keys or mount is not None and keys[mount.entity] not in entities:
                raise ValueError(f'Connection {joint.name}: {mount.entity} is not a physical entity of the scene')
        child_key = keys[joint.child.entity]
        root, free = root_body(entities[child_key])
        if not free:
            raise ValueError(f'{joint.child.entity}: attached entity requires fix_root_link=False')
        body = joint.child.body.rsplit('/', 1)[-1]
        # Converted URDF assets drop the empty "world" link the USD keeps welded to the root;
        # MuJoCo reserves that name for its world body, so it can only mean that link.
        if body not in (root, '', 'world'):
            raise ValueError(f'Connection {joint.name}: an attachment must target the root body {root!r} of {joint.child.entity}')
        parent = None
        if joint.parent is not None:
            parent_key = keys[joint.parent.entity]
            name = joint.parent.body.rsplit('/', 1)[-1] or root_body(entities[parent_key])[0]
            if entities[parent_key].spec_fn().body(name) is None:
                raise ValueError(f'Connection {joint.name}: {joint.parent.entity} has no body {name!r}')
            parent = f'{parent_key}/{name}'
        relative = joint.parent_pose.compose(inverse(joint.child_pose))
        welds.append({'name': joint.name, 'parent': parent, 'child': f'{child_key}/{root}', 'pose': relative,
                      'timeconst': 2 * dt})
    return welds


def inverse(pose):
    w, x, y, z = pose.orientation
    return Pose((0., 0., 0.), (w, -x, -y, -z)).compose(Pose(tuple(-v for v in pose.position)))


def install_welds(spec, welds):
    import mujoco
    for weld in welds:
        # relpose is body2 in body1's frame; an absent body1 is the world (environment origin).
        spec.add_equality(name=weld['name'], type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                          name1=weld['parent'] or 'world', name2=weld['child'],
                          data=[0., 0., 0., *weld['pose'].position, *weld['pose'].orientation, 1.],
                          solref=[weld['timeconst'], 1.])
        if weld['parent']:
            spec.add_exclude(bodyname1=weld['parent'], bodyname2=weld['child'])


def contact_friction(settings, keys):
    """Isaac material friction against the default friction-1 material, per entity key.

    MuJoCo has one sliding coefficient and no combine mode: the higher-priority geom's friction
    wins, so the entity's geoms get the combined dynamic friction and priority over the ground.
    Static friction has no MuJoCo counterpart."""
    from shared.backends.isaac.contact_materials import validate_materials  # SDK-free value checks
    values = settings.get('world', {}).get('contact_materials') or {}
    validate_materials(values, set(keys.values()))
    combine = COMBINE[settings.get('simulation', {}).get('physics_material', {}).get('friction_combine_mode', 'average')]
    return {entity.replace('.', '__'): combine(float(material['dynamic_friction'])) for entity, material in values.items()}


def apply_friction(spec, materials):
    for key, friction in materials.items():
        for body in spec.bodies:
            if body.name.startswith(key + '/'):
                for geom in body.geoms:
                    geom.friction = [friction, *list(geom.friction)[1:]]
                    geom.priority = max(geom.priority, 1)


HIDDEN_GEOM_GROUP = 4  # outside the camera sensor groups 0-2 (and the converter's collision group 3);
                       # record/view enable it, contacts are unaffected


def hide_camera_housings(spec, cameras):
    """Hide geoms of a camera's own body that enclose its origin (e.g. a converted D435 box).

    Isaac rasterizes with back-face culling and does not draw colliders, so a camera inside
    its housing sees out; mujoco_warp ray casts hit the housing from inside instead. The geom
    keeps its collision; it is only moved out of the camera sensors' groups, while record
    videos and the view window still draw it.
    Returns the hidden geoms as (camera, body, index on that body)."""
    import mujoco
    import numpy as np
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    hidden = []
    for name in cameras:
        camera = model.camera(name)
        body = int(camera.bodyid[0])
        point = data.cam_xpos[camera.id]
        for offset in range(model.body_geomnum[body]):
            geom = int(model.body_geomadr[body]) + offset
            if model.geom_group[geom] > 2:  # already outside the camera images
                continue
            def distance(direction):
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    return mujoco.mj_rayMesh(model, data, geom, point, direction)
                return mujoco.mju_rayGeom(data.geom_xpos[geom], data.geom_xmat[geom], model.geom_size[geom],
                                          point, direction, model.geom_type[geom])
            if all(distance(direction) >= 0 for direction in np.vstack((np.eye(3), -np.eye(3)))):
                spec.body(model.body(body).name).geoms[offset].group = HIDDEN_GEOM_GROUP
                hidden.append((name, model.body(body).name, offset))
    return hidden


def event_configuration(description, keys, entities):
    """Root resets of physical entities, then attached roots placed at their connections, as
    native reset events after mjlab's default reset."""
    from dataclasses import asdict
    from mjlab.managers import EventTermCfg
    from shared.assembly.abstract import area
    events = {}
    rules = {keys[path]: asdict(node.reset) for path, (node, _) in description.index().items()
             if path in keys and node.reset is not None}
    if rules:
        areas = {rule['region']: area(description.index()[rule['region']][0])
                 for rule in rules.values() if rule['region'] is not None}
        events['assembly_root_resets'] = EventTermCfg(func=reset_roots, mode='reset',
                                                      params={'rules': rules, 'areas': areas})
    welds = [{'parent': weld['parent'], 'child': weld['child'],
              'pose': {'position': list(weld['pose'].position), 'orientation': list(weld['pose'].orientation)}}
             for weld in connection_welds(description, entities, keys, 0.)]
    if welds:
        events['assembly_connections'] = EventTermCfg(func=reset_connections, mode='reset', params={'welds': welds})
    return events


def reset_connections(env, env_ids, *, welds):
    """Place each attached root at its joint frame with zero velocity, like the Isaac event."""
    import torch
    from shared.assembly.abstract import mounted_pose
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not len(env_ids):
        return
    for weld in welds:
        if weld['parent'] is None:
            parent = torch.zeros((len(env_ids), 7), device=env.device)
            parent[:, :3], parent[:, 3] = env.scene.env_origins[env_ids], 1.
        else:
            env.sim.forward()
            body = env.sim.mj_model.body(weld['parent']).id
            parent = torch.cat((env.sim.data.xpos[env_ids, body], env.sim.data.xquat[env_ids, body]), -1)
        state = torch.zeros((len(env_ids), 13), device=env.device)
        state[:, :7] = mounted_pose(parent[:, :3], parent[:, 3:7], weld['pose'])
        env.scene[weld['child'].split('/', 1)[0]].write_root_state_to_sim(state, env_ids=env_ids)


def reset_roots(env, env_ids, *, rules, areas):
    """Sample authored root poses like the Isaac events, with the torch RNG mjlab seeds.

    Region placement: area rotation * local-Z yaw * default orientation, zero velocity.
    States and pose_range keep the default root velocity."""
    import torch
    from shared.assembly.abstract import sample_poses
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not len(env_ids):
        return
    for name, rule in rules.items():
        entity = env.scene[name]
        state = entity.data.default_root_state[env_ids].clone()
        base = {'position': state[0, :3].tolist(), 'orientation': state[0, 3:7].tolist()}
        pose = sample_poses(rule, base, areas, len(env_ids), device=state.device, dtype=state.dtype)
        state[:, :3] = pose[:, :3] + env.scene.env_origins[env_ids]
        state[:, 3:7] = pose[:, 3:7]
        if rule['region'] is not None:
            state[:, 7:13] = 0.
        entity.write_root_state_to_sim(state, env_ids=env_ids)


def compile_scene(description, settings, factories):
    import mujoco
    from mjlab.scene import Scene
    cfg, keys = build_scene_config(description, settings, factories)
    scene = Scene(cfg, device='cpu')
    model = scene.compile()
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return scene, model, data, keys
