"""Isaac readers for authored targets and abstract entities."""
from shared.assembly.targets import target_catalog


def read_target(env, value):
    import torch
    if isinstance(value, dict):
        from shared.assembly.abstract import mounted_pose, tensor_field
        if value['commanded']:
            from .abstract_terms import term_name
            return env.command_manager.get_term(term_name(value['abstract'])).state.read(value['field'])
        data = env.scene[value['mount']].data
        pose = mounted_pose(data.root_pos_w - env.scene.env_origins, data.root_quat_w, value['pose'])
        return tensor_field(pose, value['field'])
    return torch.tensor(value, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)


def build_target(value):
    from isaaclab.managers import ObservationTermCfg
    return ObservationTermCfg(func=read_target, params={'value': value})
