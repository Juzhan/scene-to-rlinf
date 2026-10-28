"""Abstract entity poses in Isaac's native CommandManager and EventManager.

Every abstract entity that is neither mounted nor a reset region is one command
term holding a shared ``AbstractState``: the manager resamples it on episode reset
and, with ``reset.interval``, every interval seconds; the native command update
applies the pose the Task set this step (``set``). Shown abstract entities use the
same term's native debug visualization. Physical assets placed in an abstract
rectangle/box use one native reset event.
No separate stepping or reset loop exists.
"""
from functools import lru_cache


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
                self.state.pose[:] = mounted_pose(data.root_pos_w - self._env.scene.env_origins,
                                                  data.root_quat_w, self.cfg.entity['pose'])
                return
            from .python_task import task_set
            self.state.advance(self._env.step_dt, task_set(self._env, self.cfg.entity['name']))

        def _update_metrics(self):
            pass

        def _set_debug_vis_impl(self, debug_vis):
            if debug_vis and not hasattr(self, '_marker'):
                from isaaclab.markers import VisualizationMarkers
                self._marker = VisualizationMarkers(marker_config(self.cfg.entity, self.cfg.marker_path))
            if hasattr(self, '_marker'):
                self._marker.set_visibility(debug_vis)

        def _debug_vis_callback(self, event):
            from isaaclab.utils.math import quat_apply
            offsets, indices = (torch.tensor(value, device=self.device) for value in marker_parts(self.cfg.entity))
            count = len(indices)
            orientation = self.command[:, 3:7].repeat_interleave(count, 0)
            translations = (self.command[:, :3] + self._env.scene.env_origins).repeat_interleave(count, 0) + quat_apply(
                orientation, offsets.repeat(self.num_envs, 1))
            self._marker.visualize(translations=translations, orientations=orientation,
                                   marker_indices=indices.repeat(self.num_envs))

    @configclass
    class AbstractPoseCfg(CommandTermCfg):
        class_type: type = AbstractPose
        entity: dict = None
        areas: dict = None
        marker_path: str = '/Visuals/Abstract'
    return AbstractPoseCfg


def marker_parts(entity):
    """Prototype instances of one marker: (centre offsets in the entity frame, prototype indices).

    Shapes and sizes come from ``shared.assembly.abstract`` so the mjlab preview draws the same.
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


def marker_config(entity, path):
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


def term_name(name):
    return 'assembly_abstract_' + name


def attach_abstract(output, settings, *, visualize=True):
    """Command terms for commanded abstract entities and, with ``visualize``, shown ones.

    Markers exist only when a preview window/video is rendered (``visualize``), as in mjlab:
    Isaac hides them from secondary rays only, so training cameras could otherwise see them.
    """
    from shared.assembly.abstract import abstract_entities, commanded, commanded_entity, reset_areas
    from shared.assembly.targets import visualization
    shown, areas = set(visualization(settings)) if visualize else set(), reset_areas(settings)
    held = commanded(settings)
    for node in abstract_entities(settings).values():
        if node.name not in held and node.name not in shown:
            continue
        key = term_name(node.name)
        if hasattr(output['commands'], key):
            raise ValueError('Reserved abstract command name: ' + key)
        seconds = node.reset.interval if node.reset is not None and node.reset.interval else 1e9
        setattr(output['commands'], key, _config_type()(
            entity=commanded_entity(settings, node), areas=areas,
            resampling_time_range=(seconds, seconds), debug_vis=node.name in shown, marker_path='/Visuals/' + key))


def reset_root_in_regions(env, env_ids, *, rules, areas):
    """Place selected asset roots inside abstract rectangles/boxes, using torch's seeded RNG.

    Rules run after native task resets. Area pose and samples are local to each
    environment, and that environment's current origin is added once. Only
    selected environments are written. Root velocities become zero; articulation
    joint states remain owned by the task's native joint reset event.
    Orientation is area rotation * sampled local-Z yaw * asset default root orientation.
    """
    import torch
    from shared.assembly.abstract import quaternion_multiply, sample_poses

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(env_ids):
        return
    for name, rule in rules.items():
        asset = env.scene[name]
        if getattr(asset, "is_fixed_base", False):
            raise ValueError(f"Region reset cannot move a fixed-base articulation: {name}")
        state = asset.data.default_root_state[env_ids].clone()
        base = {'position': [0., 0., 0.], 'orientation': [1., 0., 0., 0.]}
        pose = sample_poses(rule, base, areas, len(env_ids), device=state.device, dtype=state.dtype)
        state[:, :3] = pose[:, :3] + env.scene.env_origins[env_ids]
        state[:, 3:7] = quaternion_multiply(pose[:, 3:7], state[:, 3:7])
        state[:, 7:13] = 0.
        asset.write_root_state_to_sim(state, env_ids=env_ids)
