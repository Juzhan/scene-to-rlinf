"""Abstract entity poses in Isaac Lab 3's native CommandManager and EventManager.

Every commanded or shown abstract entity is one command term holding a shared ``AbstractState`` (wxyz,
environment-local); physical assets placed in an abstract rectangle/box use one native reset event. Only
native poses are converted from/to xyzw here.
"""
from functools import lru_cache
from .frames import wxyz, xyzw


def term_name(name):
    return 'assembly_abstract_' + name


def marker_parts(entity):
    """Prototype instances of one marker: (centre offsets in the entity frame, prototype indices).

    Shapes and sizes come from ``shared.assembly.abstract`` so every backend's preview draws the same.
    """
    from shared.assembly.abstract import ARROW_HEAD_LENGTH, ARROW_LENGTH, marker_edges
    edges = marker_edges(entity)
    if edges is not None:
        return edges
    if entity['shape'] == 'direction':
        # Shaft, then head, along local +X.
        shaft = (1 - ARROW_HEAD_LENGTH) * ARROW_LENGTH
        return [[shaft / 2, 0., 0.], [shaft + ARROW_HEAD_LENGTH * ARROW_LENGTH / 2, 0., 0.]], [0, 1]
    return [[0., 0., 0.]], [0]


def _prototypes(entity, path):
    """Sphere point, cylinder + cone arrow direction, cylinder edges of a rectangle/box (x, y, z prototypes)."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg
    from shared.assembly.abstract import (ARROW_COLOR, ARROW_HEAD_LENGTH, ARROW_HEAD_RADIUS, ARROW_LENGTH,
                                          ARROW_WIDTH, EDGE_WIDTH, MARKER_COLOR, POINT_RADIUS)
    if entity['shape'] == 'direction':
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=ARROW_COLOR)
        return VisualizationMarkersCfg(prim_path=path, markers={
            'shaft': sim_utils.CylinderCfg(radius=ARROW_WIDTH, height=(1 - ARROW_HEAD_LENGTH) * ARROW_LENGTH,
                                           axis='X', visual_material=material),
            # A USD cone's apex is at +height/2 along its axis.
            'head': sim_utils.ConeCfg(radius=ARROW_HEAD_RADIUS * ARROW_WIDTH, height=ARROW_HEAD_LENGTH * ARROW_LENGTH,
                                      axis='X', visual_material=material)})
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=MARKER_COLOR)
    if entity['shape'] == 'point':
        return VisualizationMarkersCfg(prim_path=path, markers={
            'abstract': sim_utils.SphereCfg(radius=POINT_RADIUS, visual_material=material)})
    size = list(entity['size']) + [0.] * (3 - len(entity['size']))
    # Prototype order is the marker index: x, y, z edges.
    return VisualizationMarkersCfg(prim_path=path, markers={
        'edge_' + name: sim_utils.CylinderCfg(radius=EDGE_WIDTH / 2, height=size[axis] + EDGE_WIDTH,
                                              axis=name.upper(), visual_material=material)
        for axis, name in enumerate('xyz')})


# xyzw rotations taking a prototype's local Z onto its configured axis.
_AXIS_ROTATIONS = {'X': (0., .5 ** .5, 0., .5 ** .5), 'Y': (-.5 ** .5, 0., 0., .5 ** .5), 'Z': (0., 0., 0., 1.)}


def marker_config(entity, path):
    """The marker prototypes (_prototypes) along Z, plus each one's xyzw rotation onto its axis.

    Lab 3's Newton marker renderer builds cylinders and cones along Z and ignores ``axis``,
    so the axis is applied as a per-instance rotation instead, which every renderer draws alike."""
    config = _prototypes(entity, path)
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
            from .python_task import scene_info
            self.state = AbstractState(cfg.entity, cfg.areas, self.num_envs, self.device, package=cfg.package,
                                       scene=lambda: scene_info(env))

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
        package: str = None
        marker_path: str = '/Visuals/Abstract'
    return AbstractPoseCfg


def attach_abstract(commands, settings, *, visualize=False, package=None):
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
            entity=commanded_entity(settings, node), areas=areas, package=package,
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
