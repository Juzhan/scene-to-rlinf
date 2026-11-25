"""Read Isaac Lab 3 asset data (warp arrays exposed as torch views) by public information names."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.readers import bind_common
from .frames import wxyz

# Names/units belong to the public catalog; only native storage names live here. Root velocities
# refer to the root body's centre of mass, as in the other backends.
FIELDS = {'linear_velocity': 'root_com_lin_vel_w', 'angular_velocity': 'root_com_ang_vel_w',
          'joint_position': 'joint_pos', 'joint_velocity': 'joint_vel',
          'default_joint_position': 'default_joint_pos', 'default_joint_velocity': 'default_joint_vel',
          'joint_limits': 'joint_pos_limits'}
SUPPORTED = {*FIELDS, 'position', 'orientation', 'joint_effort', 'bounds', 'ray_hit_positions', 'frame_positions',
             'contact_force', 'contact_force_samples', 'rgb', 'distance_to_image_plane', 'distance_to_camera',
             'normals'}


def bind_reader(env, settings, path):
    # Joint columns already follow the recorded order (scene.py sets native joint_ordering).
    from isaaclab.sensors import BaseContactSensor, BaseFrameTransformer, Camera, SensorBase

    def sensor(native, field, source):
        origins = env.scene.env_origins
        if hasattr(native.cfg, 'pattern_cfg'):
            # Ray caster: its mount body's position and environment-local hits (inf where missed).
            if field == 'position':
                return lambda: native.data.pos_w.torch - origins
            if field == 'ray_hit_positions':
                return lambda: native.data.ray_hits_w.torch - origins[:, None, :]
        elif isinstance(native, BaseFrameTransformer):
            if field == 'frame_positions':
                # (N, frames, 3) relative to each environment origin.
                return lambda: native.data.target_pos_w.torch - origins[:, None, :]
        elif isinstance(native, BaseContactSensor):
            # The force acting on each selected body (ground support is +z), as on the isaac backend.
            if field == 'contact_force':
                return lambda: native.data.net_forces_w.torch
            if field == 'contact_force_samples':
                return lambda: native.data.net_forces_w_history.torch
        elif isinstance(native, Camera):
            # Raw images (N, H, W, C): uint8 RGB, depth in metres, unit normals.
            if field not in native.cfg.data_types:
                raise ValueError(f'{source}: camera does not render {field}')
            renderer = native.cfg.renderer_cfg
            if field.startswith('distance') and renderer.renderer_type == 'ovrtx':
                if renderer.depth_clipping_behavior == 'none':
                    return lambda: native.data.output[field].torch
                import torch
                # OVRTX reports inf where nothing is hit within the far plane, as Isaac does, but
                # does not apply the clipping behaviour itself.
                fill = 0. if renderer.depth_clipping_behavior == 'zero' else native.cfg.spawn.clipping_range[1]
                def clipped():
                    value = native.data.output[field].torch
                    return value.masked_fill(torch.isinf(value), fill)
                return clipped
            if field.startswith('distance') and renderer.depth_clipping_behavior == 'none':
                import torch
                def depth():
                    # Newton Warp writes 0 where nothing is hit; Isaac reports inf.
                    value = native.data.output[field].torch
                    return value.masked_fill(value <= 0, torch.inf)
                return depth
            return lambda: native.data.output[field].torch
        raise NotImplementedError(f'isaaclab3 information reader not implemented: {source}')

    def physical(source):
        if source == 'simulation.gravity':
            import torch
            value = settings.get('simulation', {}).get('gravity', [0., 0., -9.81])
            return lambda: torch.tensor(value, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
        parsed = parse_signal(source)
        if parsed is None:
            raise ValueError(f'Unknown scene information: {source}')
        key, field = parsed
        if field == 'bounds' or (field in ('position', 'orientation') and _static(env, key)):
            position, orientation, bounds = _geometry(env, key)
            value = {'position': position, 'orientation': orientation, 'bounds': bounds}[field]
            return lambda: value
        asset = env.scene[key]
        # Data properties may return new array views after a reset; read `.torch` on every call.
        # Sensors refresh outdated buffers when `.data` is read, so read it on every call too.
        if isinstance(asset, SensorBase):
            return sensor(asset, field, source)
        data = asset.data
        if field == 'position':
            return lambda: data.root_link_pos_w.torch - env.scene.env_origins
        if field == 'orientation':
            return lambda: wxyz(data.root_link_quat_w.torch)
        if field == 'joint_effort':
            # Effort the actuator models apply after saturation, like Isaac's applied_torque.
            return lambda: asset.actuators.applied_effort.torch
        if field not in FIELDS:
            raise NotImplementedError(f'isaaclab3 information reader not implemented: {source}')
        attribute = FIELDS[field]
        if not hasattr(data, attribute):
            raise ValueError(f'{source}: native asset has no {attribute}')
        return lambda: getattr(data, attribute).torch

    def sampled(value):
        from .abstract_terms import term_name
        key = term_name(value['abstract'])
        return lambda: env.command_manager.get_term(key).state.read(value['field'])
    from .robot import robots
    return bind_common(env, settings, path, bind_physical=physical, bind_sampled_target=sampled,
                       robots=robots(env, settings))


def _static(env, key):
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    config = getattr(env.scene.cfg, key, None)
    return isinstance(config, AssetBaseCfg) and not isinstance(config, (ArticulationCfg, RigidObjectCfg))


def _geometry(env, key):
    """``(position, orientation, bounds)`` of entity ``key`` as spawned, batched over environments.

    The frame is the one the root readings use: the root body of a rigid object or articulation
    (the body no joint of the asset hangs from), the entity prim of a static one. ``bounds`` are the
    corners of the drawn geometry in that frame, at the default joint positions; the pose of a static
    entity is constant. Read once from the stage, from one environment's copy when each has one.
    """
    cache = env.__dict__.setdefault('_assembly_geometry', {})
    if key in cache:
        return cache[key]
    import re
    import torch
    from isaaclab.sim.utils.queries import find_first_matching_prim
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    stage = env.sim.stage
    template = getattr(env.scene.cfg, key).prim_path
    prim = stage.GetPrimAtPath(template) if Sdf.Path.IsValidPathString(template) else Usd.Prim()
    clone = None if prim.IsValid() else find_first_matching_prim(template, stage)
    if clone is not None:
        prim = clone  # one environment's copy; the others are clones at their origins
    if not prim.IsValid():
        raise ValueError(f'{key}: no prim at {template}')
    number = re.search(r'/env_(\d+)/', str(prim.GetPath()) + '/') if clone is not None else None
    root = prim
    if not _static(env, key):
        prims = list(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
        bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        children = {target for p in prims if p.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(p).GetBody0Rel().GetTargets()
                    for target in UsdPhysics.Joint(p).GetBody1Rel().GetTargets()}
        root = next((body for body in bodies if body.GetPath() not in children), prim)
    time = Usd.TimeCode.Default()
    transform = Gf.Transform(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(time))
    frame = Gf.Matrix4d().SetRotate(transform.GetRotation()) * Gf.Matrix4d().SetTranslate(transform.GetTranslation())
    box = UsdGeom.BBoxCache(time, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]).ComputeWorldBound(prim)
    box.Transform(frame.GetInverse())
    extent = box.ComputeAlignedRange()
    if extent.IsEmpty():
        raise ValueError(f'{key}: no drawn geometry to bound')
    device = env.device
    origins = env.scene.env_origins
    position = torch.tensor(list(transform.GetTranslation()), device=device, dtype=torch.float32)
    position = position - (origins[int(number.group(1))] if number else origins)
    quaternion = transform.GetRotation().GetQuat()
    orientation = torch.tensor([quaternion.GetReal(), *quaternion.GetImaginary()], device=device, dtype=torch.float32)
    bounds = torch.tensor([list(extent.GetMin()), list(extent.GetMax())], device=device, dtype=torch.float32)
    cache[key] = (position.expand(env.num_envs, 3), orientation.expand(env.num_envs, 4),
                  bounds.expand(env.num_envs, 2, 3))
    return cache[key]
