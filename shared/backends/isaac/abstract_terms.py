"""Abstract entity poses in Isaac's native CommandManager and EventManager.

A randomized abstract entity is one command term: the manager resamples it on
episode reset and, with ``reset.interval``, every interval seconds. Shown
abstract entities use the same term's native debug visualization. Physical
assets placed in an abstract rectangle/box use one native reset event.
No separate stepping or reset loop exists.
"""
from functools import lru_cache


@lru_cache(maxsize=1)
def _config_type():
    import torch
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass
    from shared.assembly.abstract import mounted_pose, sample_poses

    class AbstractPose(CommandTerm):
        """(N, 7) environment-local xyz + wxyz pose of one abstract entity."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            base = cfg.entity['pose']
            self._command = torch.tensor([*base['position'], *base['orientation']],
                                         device=self.device).repeat(self.num_envs, 1)

        @property
        def command(self):
            return self._command

        def _resample_command(self, env_ids):
            if self.cfg.entity['reset'] is not None:
                self._command[env_ids] = sample_poses(self.cfg.entity['reset'], self.cfg.entity['pose'],
                                                      self.cfg.areas, len(env_ids), device=self.device)

        def _update_command(self):
            if self.cfg.entity['mount'] is not None:
                data = self._env.scene[self.cfg.entity['mount']].data
                self._command[:] = mounted_pose(data.root_pos_w - self._env.scene.env_origins,
                                                data.root_quat_w, self.cfg.entity['pose'])

        def _update_metrics(self):
            pass

        def _set_debug_vis_impl(self, debug_vis):
            if debug_vis and not hasattr(self, '_marker'):
                from isaaclab.markers import VisualizationMarkers
                self._marker = VisualizationMarkers(marker_config(self.cfg.entity, self.cfg.marker_path))
            if hasattr(self, '_marker'):
                self._marker.set_visibility(debug_vis)

        def _debug_vis_callback(self, event):
            position = self.command[:, :3] + self._env.scene.env_origins
            orientation = self.command[:, 3:7]
            edges = marker_edges(self.cfg.entity)
            if edges is None:
                self._marker.visualize(translations=position, orientations=orientation)
                return
            from isaaclab.utils.math import quat_apply
            offsets, indices = (torch.tensor(value, device=self.device) for value in edges)
            count = len(indices)
            orientation = orientation.repeat_interleave(count, 0)
            translations = position.repeat_interleave(count, 0) + quat_apply(
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


EDGE_WIDTH = 0.008


def marker_edges(entity):
    """Edge centres in the entity frame and their axis prototype, or None for solid markers.

    Rectangles and boxes are drawn as wireframes: RTX real-time rendering
    ignores surface opacity, so a solid volume would hide the robot inside it.
    """
    if entity['shape'] not in ('rectangle', 'box'):
        return None
    size = list(entity['size']) + [0.] * (3 - len(entity['size']))
    offsets, indices = [], []
    for axis in range(3):
        if not size[axis]:
            continue
        others = [other for other in range(3) if other != axis]
        corners = [(a, b) for a in ((-.5, .5) if size[others[0]] else (0.,))
                   for b in ((-.5, .5) if size[others[1]] else (0.,))]
        for a, b in corners:
            offset = [0.] * 3
            offset[others[0]], offset[others[1]] = a * size[others[0]], b * size[others[1]]
            offsets.append(offset)
            indices.append(axis)
    return offsets, indices


def marker_config(entity, path):
    """Sphere point, arrow direction, wireframe rectangle/box."""
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkersCfg
    from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
    if entity['shape'] == 'direction':
        return GREEN_ARROW_X_MARKER_CFG.replace(prim_path=path)
    material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0))
    if entity['shape'] == 'point':
        return VisualizationMarkersCfg(prim_path=path, markers={
            'abstract': sim_utils.SphereCfg(radius=0.04, visual_material=material)})
    size = list(entity['size']) + [0.] * (3 - len(entity['size']))
    markers = {}
    for axis in range(3):
        edge = [EDGE_WIDTH] * 3
        edge[axis] = size[axis] + EDGE_WIDTH
        # Prototype order is the marker index: x, y, z edges.
        markers['edge_' + 'xyz'[axis]] = sim_utils.CuboidCfg(size=tuple(edge), visual_material=material)
    return VisualizationMarkersCfg(prim_path=path, markers=markers)


def term_name(name):
    return 'assembly_abstract_' + name


def attach_abstract(output, settings):
    """Command terms for randomized or shown abstract entities."""
    from dataclasses import asdict
    from shared.assembly.abstract import abstract_entities, entity_key, reset_areas
    from shared.assembly.targets import visualization
    shown, areas = set(visualization(settings)), reset_areas(settings)
    for node in abstract_entities(settings).values():
        if node.reset is None and node.name not in shown:
            continue
        key = term_name(node.name)
        if hasattr(output['commands'], key):
            raise ValueError('Reserved abstract command name: ' + key)
        seconds = node.reset.interval if node.reset is not None and node.reset.interval else 1e9
        entity = {'shape': node.shape, 'size': list(node.size), 'pose': asdict(node.pose),
                  'reset': None if node.reset is None else asdict(node.reset),
                  'mount': None if node.mount is None else entity_key(settings, node.mount.entity)}
        setattr(output['commands'], key, _config_type()(
            entity=entity, areas=areas, resampling_time_range=(seconds, seconds),
            debug_vis=node.name in shown, marker_path='/Visuals/' + key))


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
