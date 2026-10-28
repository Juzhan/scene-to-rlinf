"""Read Isaac Lab 3 asset data (warp arrays exposed as torch views) by public information names."""
from shared.assembly.entity_information import parse_signal
from shared.assembly.readers import bind_common
from .frames import wxyz

# Names/units belong to the public catalog; only native storage names live here. Root velocities
# refer to the root body's centre of mass, as in the other backends.
FIELDS = {'linear_velocity': 'root_com_lin_vel_w', 'angular_velocity': 'root_com_ang_vel_w',
          'joint_position': 'joint_pos', 'joint_velocity': 'joint_vel',
          'default_joint_position': 'default_joint_pos', 'default_joint_velocity': 'default_joint_vel'}
SUPPORTED = {*FIELDS, 'position', 'orientation', 'joint_effort', 'ray_hit_positions', 'frame_positions',
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
