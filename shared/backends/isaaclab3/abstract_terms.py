"""Abstract entity poses in Isaac Lab 3's native CommandManager and EventManager.

As in the ``isaac`` backend: every commanded or shown abstract entity is one command term holding a
shared ``AbstractState`` (wxyz, environment-local); physical assets placed in an abstract
rectangle/box use one native reset event. Only native poses are converted from/to xyzw here.
"""
from functools import lru_cache
from shared.backends.isaac.abstract_terms import marker_config as isaac_marker_config, marker_parts, term_name
from .frames import wxyz, xyzw


# xyzw rotations taking a prototype's local Z onto its configured axis.
_AXIS_ROTATIONS = {'X': (0., .5 ** .5, 0., .5 ** .5), 'Y': (-.5 ** .5, 0., 0., .5 ** .5), 'Z': (0., 0., 0., 1.)}


def marker_config(entity, path):
    """The isaac backend's marker prototypes along Z, plus each one's xyzw rotation onto its axis.

    Lab 3's Newton marker renderer builds cylinders and cones along Z and ignores ``axis``,
    so the axis is applied as a per-instance rotation instead, which every renderer draws alike."""
    config = isaac_marker_config(entity, path)
    rotations = []
    for prototype in config.markers.values():
        rotations.append(_AXIS_ROTATIONS[getattr(prototype, 'axis', 'Z')])
        if hasattr(prototype, 'axis'):
            prototype.axis = 'Z'
    return config, rotations


@lru_cache(maxsize=1)
def _config_type():
    import torch
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass
    from shared.assembly.abstract import AbstractState, mounted_pose

    class AbstractPose(CommandTerm):
        """(N, 7) environment-local xyz + wxyz pose of one abstract entity."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.state = AbstractState(cfg.entity, cfg.areas, self.num_envs, self.device)

        @property
        def command(self):
            return self.state.pose

        def _resample_command(self, env_ids):
            if self.cfg.entity['mount'] is None:
                self.state.resample(env_ids)

        def _update_command(self):
            if self.cfg.entity['mount'] is not None:
                data = self._env.scene[self.cfg.entity['mount']].data
                self.state.pose[:] = mounted_pose(data.root_link_pos_w.torch - self._env.scene.env_origins,
                                                  wxyz(data.root_link_quat_w.torch), self.cfg.entity['pose'])
                return
            from .python_task import task_set
            self.state.advance(self._env.step_dt, task_set(self._env, self.cfg.entity['name']))

        def _update_metrics(self):
            pass

        def _set_debug_vis_impl(self, debug_vis):
            if debug_vis and not hasattr(self, '_marker'):
                from isaaclab.markers import VisualizationMarkers
                config, rotations = marker_config(self.cfg.entity, self.cfg.marker_path)
                self._marker = VisualizationMarkers(config)
                self._rotations = torch.tensor(rotations, device=self.device)
            if hasattr(self, '_marker'):
                self._marker.set_visibility(debug_vis)

        def _debug_vis_callback(self, event):
            from isaaclab.utils.math import quat_apply, quat_mul
            offsets, indices = (torch.tensor(value, device=self.device) for value in marker_parts(self.cfg.entity))
            count = len(indices)
            orientation = xyzw(self.command[:, 3:7]).repeat_interleave(count, 0)
            translations = (self.command[:, :3] + self._env.scene.env_origins).repeat_interleave(count, 0) + quat_apply(
                orientation, offsets.repeat(self.num_envs, 1))
            indices = indices.repeat(self.num_envs)
            self._marker.visualize(translations=translations, orientations=quat_mul(orientation, self._rotations[indices]),
                                   marker_indices=indices)

    @configclass
    class AbstractPoseCfg(CommandTermCfg):
        class_type: type = AbstractPose
        entity: dict = None
        areas: dict = None
        marker_path: str = '/Visuals/Abstract'
    return AbstractPoseCfg


def attach_abstract(commands, settings, *, visualize=False):
    """Command terms for commanded abstract entities and, with ``visualize``, shown ones."""
    from shared.assembly.abstract import abstract_entities, commanded, commanded_entity, reset_areas
    from shared.assembly.targets import visualization
    shown, areas = set(visualization(settings)) if visualize else set(), reset_areas(settings)
    held = commanded(settings)
    for node in abstract_entities(settings).values():
        if node.name not in held and node.name not in shown:
            continue
        key = term_name(node.name)
        if key in commands:
            raise ValueError('Reserved abstract command name: ' + key)
        seconds = node.reset.interval if node.reset is not None and node.reset.interval else 1e9
        commands[key] = _config_type()(
            entity=commanded_entity(settings, node), areas=areas,
            resampling_time_range=(seconds, seconds), debug_vis=node.name in shown, marker_path='/Visuals/' + key)


def reset_root_in_regions(env, env_ids, *, rules, areas):
    """Place selected asset roots inside abstract rectangles/boxes, using torch's seeded RNG.

    Orientation is area rotation * sampled local-Z yaw * asset default root orientation;
    root velocities become zero and joint states stay with the other reset events.
    """
    import torch
    from shared.assembly.abstract import quaternion_multiply, sample_poses

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(env_ids):
        return
    base = {'position': [0., 0., 0.], 'orientation': [1., 0., 0., 0.]}
    for name, rule in rules.items():
        asset = env.scene[name]
        if getattr(asset, 'is_fixed_base', False):
            raise ValueError(f'Region reset cannot move a fixed-base articulation: {name}')
        default = asset.data.default_root_pose.torch[env_ids]
        pose = sample_poses(rule, base, areas, len(env_ids), device=default.device, dtype=default.dtype)
        root = torch.cat((pose[:, :3] + env.scene.env_origins[env_ids],
                          xyzw(quaternion_multiply(pose[:, 3:7], wxyz(default[:, 3:7])))), -1)
        asset.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(env_ids), 6, device=env.device),
                                               env_ids=env_ids)
