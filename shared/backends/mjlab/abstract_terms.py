"""Commanded and shown abstract entities as mjlab native command terms.

Every abstract entity that is neither mounted nor a reset region is one command term
holding a shared ``AbstractState``, as in Isaac: the manager resamples it on episode reset
and every ``reset.interval``; the native command update applies the pose the Task set this
step (``set``). Each entity in ``visualization.abstract`` is drawn by the term's
``_debug_vis_impl`` through mjlab's DebugVisualizer, only for the preview window/video;
the scene cameras (mujoco_warp) never render them. Shapes, sizes and colours come from
``shared.assembly.abstract``.
"""
from functools import lru_cache


def draw_marker(visualizer, entity, position, rotation):
    """One entity at world ``position`` with 3x3 ``rotation`` (numpy), in shared marker sizes."""
    import numpy as np
    from shared.assembly.abstract import (ARROW_COLOR, ARROW_LENGTH, ARROW_WIDTH, EDGE_WIDTH, MARKER_COLOR,
                                          POINT_RADIUS, marker_edges)
    color = (*MARKER_COLOR, 1.)
    edges = marker_edges(entity)
    if edges is not None:
        size = list(entity['size']) + [0.] * (3 - len(entity['size']))
        for offset, axis in zip(*edges):
            centre = position + rotation @ np.asarray(offset)
            half = rotation[:, axis] * (size[axis] + EDGE_WIDTH) / 2
            visualizer.add_cylinder(centre - half, centre + half, EDGE_WIDTH / 2, color)
    elif entity['shape'] == 'direction':
        # MuJoCo 3.5 draws an arrow connector over half of the from-to distance.
        visualizer.add_arrow(position, position + 2 * ARROW_LENGTH * rotation[:, 0], (*ARROW_COLOR, 1.),
                             width=ARROW_WIDTH)
    else:
        visualizer.add_sphere(position, POINT_RADIUS, color)


@lru_cache(maxsize=1)
def _config_type():
    from dataclasses import dataclass
    import torch
    from mjlab.managers import CommandTerm, CommandTermCfg
    from shared.assembly.abstract import AbstractState, mounted_pose

    class AbstractPose(CommandTerm):
        """(N, 7) environment-local xyz + wxyz pose of one abstract entity."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.state = AbstractState(cfg.entity, cfg.areas, self.num_envs, self.device)

        @property
        def command(self):
            if self.cfg.entity['mount'] is None:
                return self.state.pose
            data = self._env.scene[self.cfg.entity['mount']].data
            return mounted_pose(data.root_link_pos_w - self._env.scene.env_origins, data.root_link_quat_w,
                                self.cfg.entity['pose'])

        def _resample_command(self, env_ids):
            if self.cfg.entity['mount'] is None:
                self.state.resample(env_ids)

        def _update_command(self, env_ids=None):
            # mjlab >= 1.6 also calls this from reset() with the reset ids; those rows
            # are fresh, and only the per-step update (None) applies the Task set.
            if self.cfg.entity['mount'] is not None or env_ids is not None:
                return
            from .python_task import task_set
            self.state.advance(self._env.step_dt, task_set(self._env, self.cfg.entity['name']))

        def _update_metrics(self):
            pass

        def _debug_vis_impl(self, visualizer):
            import mujoco
            import numpy as np
            indices = list(visualizer.get_env_indices(self.num_envs))
            if not indices:
                return
            pose = self.command[indices].double().cpu().numpy()
            origins = self._env.scene.env_origins[indices].double().cpu().numpy()
            rotation = np.empty(9)
            for local, origin in zip(pose, origins):
                mujoco.mju_quat2Mat(rotation, local[3:7])
                draw_marker(visualizer, self.cfg.entity, local[:3] + origin, rotation.reshape(3, 3))

    @dataclass(kw_only=True)
    class AbstractPoseCfg(CommandTermCfg):
        entity: dict
        areas: dict

        def build(self, env):
            return AbstractPose(self, env)
    return AbstractPoseCfg


def term_name(name):
    return 'assembly_abstract_' + name


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
            resampling_time_range=(seconds, seconds), debug_vis=node.name in shown)


def guarded_visualizer(scene, model):
    """mjlab's native visualizer on ``scene`` for environment 0, refusing to write past ``scene.maxgeom``.

    The native visualizer increments ``ngeom`` without a bound check."""
    from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

    class Guarded(MujocoNativeDebugVisualizer):
        def _room(self):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError(f'Preview scene is full ({scene.maxgeom} geoms); '
                                   'too many debug markers (abstract entities, sensor rays)')

        def add_arrow(self, *args, **kwargs):
            self._room()
            super().add_arrow(*args, **kwargs)

        def add_sphere(self, *args, **kwargs):
            self._room()
            super().add_sphere(*args, **kwargs)

        def add_cylinder(self, *args, **kwargs):
            self._room()
            super().add_cylinder(*args, **kwargs)
    return Guarded(scene, model, 0)
