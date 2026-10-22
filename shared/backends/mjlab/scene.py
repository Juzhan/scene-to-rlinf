"""Shared scene -> native mjlab SceneCfg. No policy or training loop."""
from math import isfinite
from shared.assembly.spec import Abstract, Pose

INITIAL_STATE = {'joint_positions', 'joint_velocities', 'linear_velocity', 'angular_velocity'}


def validate(description, settings):
    # Fail before starting the simulator rather than silently ignoring features.
    from .assets import SENSORS
    allowed = {'schema_version', 'name', 'entities', 'connections', 'simulation',
               'world', 'authoring', 'control', 'targets', 'application', 'observations', 'visualization'}
    extra = settings.keys() - allowed
    if extra:
        raise NotImplementedError(f'mjlab scene-only backend does not yet implement: {sorted(extra)}')
    sim = settings.get('simulation', {})
    # render_interval only paces Isaac's renderer; mjlab renders on request.
    if sim.keys() - {'dt', 'gravity', 'render_interval', 'physics_material'}:
        raise NotImplementedError('mjlab scene simulation supports only dt, gravity, render_interval and physics_material')
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
    if world.keys() - {'env_spacing', 'contact_materials'} or world.get('contact_materials'):
        raise NotImplementedError('mjlab scene world supports only env_spacing and empty contact_materials')
    if description.ordered_connections():
        raise NotImplementedError('mjlab physical connections are not implemented yet')
    if settings.get('visualization', {}).get('abstract'):
        raise NotImplementedError('mjlab abstract entity visualization is not connected yet')
    for path, (node, _) in description.index().items():
        if isinstance(node, Abstract):
            # Static and mounted abstract entities are read through shared readers.
            if node.reset is not None:
                raise NotImplementedError(f'{path}: mjlab randomized abstract entities (reset) are not connected yet; '
                                          'static and mounted abstract entities are readable')
            continue
        # prim_name only renames the Isaac prim; mjlab keys follow the logical path.
        if node.frames or node.meshes:
            raise NotImplementedError(f'{path}: mjlab body frames and ray meshes are not implemented yet')
        if node.mount and node.factory not in SENSORS:
            raise NotImplementedError(f'{path}: mjlab mounts only sensors {list(SENSORS)}; physical attachment is not implemented')
        if node.reset is not None and node.factory in SENSORS:
            raise ValueError(f'{path}: sensors cannot be reset')
        if node.initial_state.keys() - INITIAL_STATE:
            raise NotImplementedError(f'{path}: mjlab initial_state supports only {sorted(INITIAL_STATE)}')


def build_scene_config(description, settings, factories):
    from dataclasses import replace
    import mujoco
    from mjlab.sensor import CameraSensorCfg, SensorCfg
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
        elif hasattr(cfg, 'target_body'):
            if mount is None:
                raise ValueError(f'{path}: frame targets are relative to a mount entity')
            cfg = replace(cfg, name=key, entity=keys[mount.entity])
        else:
            raise NotImplementedError(f'{path}: mjlab sensor {type(cfg).__name__} is not connected')
        native_sensors.append(cfg)
    sim = settings.get('simulation', {})
    mounted = [cfg.name for cfg in native_sensors if isinstance(cfg, CameraSensorCfg) and cfg.parent_body]
    def configure(spec):
        spec.option.timestep = sim.get('dt', .005)
        spec.option.gravity = sim.get('gravity', [0., 0., -9.81])
        if mounted:
            hide_camera_housings(spec, mounted)
    return SceneCfg(num_envs=1, env_spacing=settings.get('world', {}).get('env_spacing', 4.),
                    entities=entities, sensors=tuple(native_sensors), spec_fn=configure), keys


HIDDEN_GEOM_GROUP = 3  # outside the default camera/viewer groups 0-2; contacts are unaffected


def hide_camera_housings(spec, cameras):
    """Hide geoms of a camera's own body that enclose its origin (e.g. a converted D435 box).

    Isaac rasterizes with back-face culling and does not draw colliders, so a camera inside
    its housing sees out; mujoco_warp ray casts hit the housing from inside instead. The geom
    keeps its collision; it is only moved out of the rendered groups (also in record videos).
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
            def distance(direction):
                if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    return mujoco.mj_rayMesh(model, data, geom, point, direction)
                return mujoco.mju_rayGeom(data.geom_xpos[geom], data.geom_xmat[geom], model.geom_size[geom],
                                          point, direction, model.geom_type[geom])
            if all(distance(direction) >= 0 for direction in np.vstack((np.eye(3), -np.eye(3)))):
                spec.body(model.body(body).name).geoms[offset].group = HIDDEN_GEOM_GROUP
                hidden.append((name, model.body(body).name, offset))
    return hidden


def event_configuration(description, keys):
    """Root resets of physical entities as one native reset event after mjlab's default reset."""
    from dataclasses import asdict
    from mjlab.managers import EventTermCfg
    from shared.assembly.abstract import area
    rules = {keys[path]: asdict(node.reset) for path, (node, _) in description.index().items()
             if path in keys and node.reset is not None}
    if not rules:
        return {}
    areas = {rule['region']: area(description.index()[rule['region']][0])
             for rule in rules.values() if rule['region'] is not None}
    return {'assembly_root_resets': EventTermCfg(func=reset_roots, mode='reset',
                                                 params={'rules': rules, 'areas': areas})}


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
