"""Read native batched mjlab EntityData using the public information names."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.joint_order import JOINT_FIELDS
from shared.assembly.readers import bind_common

# Names/units belong to the public catalog; only native storage names live here.
# Isaac's root_lin_vel_w/root_ang_vel_w describe the root body's centre of mass.
# Keep the same reference point here; the root-link origin can have another velocity.
FIELDS = {'orientation': 'root_link_quat_w', 'linear_velocity': 'root_com_lin_vel_w',
          'angular_velocity': 'root_com_ang_vel_w', 'joint_position': 'joint_pos',
          'joint_velocity': 'joint_vel', 'default_joint_position': 'default_joint_pos',
          'default_joint_velocity': 'default_joint_vel'}
SENSOR_FIELDS = {'frame_positions', 'rgb', 'distance_to_image_plane', 'distance_to_camera',
                 'contact_force', 'contact_force_samples', 'ray_hit_positions'}
SUPPORTED = {*FIELDS, *SENSOR_FIELDS, 'position', 'joint_effort'}


def image_plane_cosine(cfg, device):
    """cos of each pixel ray to the optical axis: image-plane depth = ray distance * cos."""
    import math
    import torch
    tangent = math.tan(math.radians(cfg.fovy) / 2)
    u = (2 * (torch.arange(cfg.width, device=device) + .5) / cfg.width - 1) * tangent * cfg.width / cfg.height
    v = (1 - 2 * (torch.arange(cfg.height, device=device) + .5) / cfg.height) * tangent
    return (1 + u[None, :] ** 2 + v[:, None] ** 2).rsqrt()[..., None]


def depth(distance, cfg, kind, cosine=None):
    """mujoco_warp ray distance (0 = miss) -> Isaac depth with its far-plane clipping behaviour."""
    import torch
    value = distance * cosine if kind == 'distance_to_image_plane' else distance.clone()
    value[(distance <= 0) | (value > cfg.clipping_range[1])] = torch.inf
    if cfg.depth_clipping != 'none':
        value[torch.isinf(value)] = 0. if cfg.depth_clipping == 'zero' else cfg.clipping_range[1]
    return value


def bind_reader(env, settings, path):
    def sensor(key, field, source):
        native = env.scene[key]
        cfg = native.cfg if hasattr(native, 'cfg') else None
        if field == 'ray_hit_positions':
            if not hasattr(cfg, 'pattern'):
                raise ValueError(f'{source}: {key} is not a ray sensor')
            # (N, rays, 3) relative to each environment origin, like Isaac's ray_hits_w minus env origins.
            # The rays come from assets.isaac_grid in Isaac's grid order, so no reordering is needed.
            # mjlab reports a miss as distance -1 with hit_pos_w at the ray start; Isaac reports inf.
            def hits():
                import torch
                data = native.data
                return torch.where((data.distances < 0)[..., None], torch.inf,
                                   data.hit_pos_w - env.scene.env_origins[:, None, :])
            return hits
        if field == 'frame_positions':
            if not hasattr(native, 'frame'):
                raise ValueError(f'{source}: {key} is not a frame_transformer')
            # One measured frame, (N, frames, 3) relative to each environment origin.
            return lambda: (native.data - env.scene.env_origins)[:, None, :]
        if field in ('contact_force', 'contact_force_samples'):
            if not hasattr(cfg, 'primary'):
                raise ValueError(f'{source}: {key} is not a contact sensor')
            # MuJoCo's net force is the one the body exerts on the others; Isaac reports the force
            # acting on the body (ground support is +z).
            if field == 'contact_force':
                return lambda: -native.data.force
            if not cfg.history_length:
                return lambda: -native.data.force[:, None]
            # mjlab keeps (N, bodies, samples, 3); Isaac's history is samples before bodies.
            return lambda: -native.data.force_history.transpose(1, 2)
        if cfg is None or not hasattr(cfg, 'depth_clipping'):
            raise ValueError(f'{source}: {key} is not a camera')
        if field == 'rgb':
            if 'rgb' not in cfg.data_types:
                raise ValueError(f'{source}: camera does not render rgb')
            return lambda: native.data.rgb
        if 'depth' not in cfg.data_types:
            raise ValueError(f'{source}: camera does not render depth')
        cosine = image_plane_cosine(cfg, env.device) if field == 'distance_to_image_plane' else None
        return lambda: depth(native.data.depth, cfg, field, cosine)

    def physical(source):
        read = native(source)
        parsed = parse_signal(source)
        if parsed and parsed[1] in JOINT_FIELDS:
            index = joint_index(env, parsed[0])
            if index is not None:
                return lambda: read()[:, index]
        return read

    def native(source):
        if source == 'simulation.gravity':
            import torch
            value = settings.get('simulation', {}).get('gravity', [0., 0., -9.81])
            return lambda: torch.tensor(value, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
        parsed = parse_signal(source)
        if parsed is None:
            raise ValueError(f'Unknown scene information: {source}')
        key, field = parsed
        if field in SENSOR_FIELDS:
            return sensor(key, field, source)
        entity = env.scene[key]
        if field == 'position' and hasattr(getattr(entity, 'cfg', None), 'pattern'):
            # Ray sensor: Isaac's RayCasterData.pos_w is the mount body origin (the sensor pose only
            # offsets the ray starts); mjlab's pos_w is the same frame body position.
            return lambda: entity.data.pos_w - env.scene.env_origins
        data = entity.data
        if field == 'position':
            return lambda: data.root_link_pos_w - env.scene.env_origins
        if field == 'joint_effort':
            # Generalized actuator force on each joint DOF, like Isaac's applied_torque.
            dofs = entity.indexing.joint_v_adr
            return lambda: data.data.qfrc_actuator[:, dofs]
        if field not in FIELDS:
            raise NotImplementedError(f'mjlab information reader not implemented: {source}')
        attribute = FIELDS[field]
        if not hasattr(data, attribute):
            raise ValueError(f'{source}: native entity has no {attribute}')
        return lambda: getattr(data, attribute)
    def sampled(value):
        from .abstract_terms import term_name
        key = term_name(value['abstract'])
        return lambda: env.command_manager.get_term(key).state.read(value['field'])
    from .robot import joint_index, robots
    return bind_common(env, settings, path, bind_physical=physical, bind_sampled_target=sampled,
                       robots=robots(env, settings))
