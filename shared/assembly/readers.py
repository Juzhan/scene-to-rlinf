"""Bind common scene information; backends provide only physical tensor readers."""
import torch
from .targets import target_catalog
from .abstract import mounted_pose, tensor_field
from .robot import FRAME_FIELDS, frame_observations


def bind_common(env, settings, path, *, bind_physical, bind_sampled_target, robots=None):
    targets = target_catalog(settings)
    frame = frame_observations(settings.get('_robot_frames') or {}).get(path)
    if frame is not None:
        # A named robot frame is read through the backend's robot interface, like frame_pose.
        robot, name, field = frame
        kinematics = robots(robot)
        index = FRAME_FIELDS.index(field)
        return lambda: kinematics.frame_pose(name)[index]
    if path in targets:
        value = targets[path][0]
        if isinstance(value, dict) and value['sampled']:
            return bind_sampled_target(value)
        if isinstance(value, dict):
            # A mounted abstract entity follows its entity's root; this is a
            # reading only, composed from the backend's physical root readers.
            position = bind_physical(value['mount'] + '.position')
            orientation = bind_physical(value['mount'] + '.orientation')
            return lambda: tensor_field(mounted_pose(position(), orientation(), value['pose']), value['field'])
        return lambda: torch.tensor(value, device=env.device, dtype=torch.float32).expand(env.num_envs, -1)
    if path == 'simulation.step_dt':
        return lambda: torch.full((env.num_envs, 1), env.step_dt, device=env.device)
    if path == 'simulation.time':
        return lambda: env.episode_length_buf[:, None] * env.step_dt
    if path.split('.')[0] in settings.get('control', {}).get('components', {}):
        role, field, *suffix = path.split('.')
        if field not in ('input', 'output', 'previous_output'):
            raise ValueError(f'Unknown component information: {path}')
        if suffix:
            if field != 'input':
                raise ValueError(f'Component output is a tensor: {path}')
            return lambda: env._assembly_control_chain.read(role, 'input')['.'.join(suffix)]
        return lambda: env._assembly_control_chain.read(role, field)
    return bind_physical(path)
