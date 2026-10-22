"""Isaac readers for public entity information."""
from shared.assembly.entity_information import (SIGNALS, signal_information, command_information, signal_profile, signal_id, parse_signal)


def primitive(env, path):
    """Read stored backend data; only normalize the environment origin convention."""
    import torch
    if path == 'simulation.gravity':
        return torch.tensor(env.cfg.sim.gravity, device=env.device).expand(env.num_envs, -1)
    key, field = parse_signal(path)
    data = env.scene[key].data
    attributes = {'default_joint_position': 'default_joint_pos', 'default_joint_velocity': 'default_joint_vel',
                  'ray_hit_positions': 'ray_hits_w', 'frame_positions': 'target_pos_w',
                  'contact_force': 'net_forces_w', 'contact_force_samples': 'net_forces_w_history'}
    if field == 'position':
        value = data.root_pos_w if hasattr(data, 'root_pos_w') else data.pos_w
        return value - env.scene.env_origins
    value = getattr(data, attributes[field])
    if field in ('ray_hit_positions', 'frame_positions'):
        value = value - env.scene.env_origins[:, None, :]
    return value


def build_signal(name):
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    direct = {'default_joint_position', 'default_joint_velocity', 'ray_hit_positions', 'frame_positions',
              'contact_force', 'contact_force_samples', 'position'}
    if name == 'simulation.gravity' or parse_signal(name)[1] in direct:
        return ObservationTermCfg(func=primitive, params={'path': name})
    key, signal = parse_signal(name)
    label, function, capability, units, dimension = SIGNALS[signal]
    parameter = 'sensor_cfg' if capability in ('height_scan', 'contact_forces', 'image', 'frame_transform') else 'asset_cfg'
    params = {parameter: SceneEntityCfg(key)}
    if capability == 'image':
        # Raw sensor values: uint8 RGB, depth in metres, unit normals; scaling belongs to the model.
        params.update(data_type=signal, normalize=False)
    return ObservationTermCfg(func=getattr(mdp, function), params=params)
