"""Read native batched mjlab EntityData using the public information names."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.readers import bind_common

# Names/units belong to the public catalog; only native storage names live here.
# Isaac's root_lin_vel_w/root_ang_vel_w describe the root body's centre of mass.
# Keep the same reference point here; the root-link origin can have another velocity.
FIELDS = {'orientation': 'root_link_quat_w', 'linear_velocity': 'root_com_lin_vel_w',
          'angular_velocity': 'root_com_ang_vel_w', 'joint_position': 'joint_pos',
          'joint_velocity': 'joint_vel', 'default_joint_position': 'default_joint_pos',
          'default_joint_velocity': 'default_joint_vel'}
SENSOR_FIELDS = {'frame_positions', 'rgb', 'distance_to_image_plane', 'distance_to_camera'}
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
        if field == 'frame_positions':
            if not hasattr(native, 'frame'):
                raise ValueError(f'{source}: {key} is not a frame_transformer')
            # One measured frame, (N, frames, 3) relative to each environment origin.
            return lambda: (native.data - env.scene.env_origins)[:, None, :]
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
        raise NotImplementedError('mjlab randomized abstract entities are not connected yet')
    return bind_common(env, settings, path, bind_physical=physical, bind_sampled_target=sampled)
