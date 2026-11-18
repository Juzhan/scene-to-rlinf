"""Kinematic entities on Isaac Lab 3 (see ``shared/assembly/kinematic.py``).

The root becomes a UsdPhysics kinematic rigid body, which Newton imports as a kinematic body (a
MuJoCo-Warp mocap-like free joint) and PhysX as a kinematic actor. One native command term per
entity writes the pose the Task set, else the pose of its scene ``motion`` (``shared/assembly/motion.py``),
in the command update after resets and before observations.
"""
from functools import lru_cache
from .frames import wxyz, xyzw


def make_kinematic(native, path):
    """The entity's rigid object config with a kinematic root; static props become rigid objects."""
    import isaaclab.sim as sim
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
    from isaaclab.sim.spawners.utils import fragment_mapping
    spawn = getattr(native, 'spawn', None)
    if isinstance(native, ArticulationCfg) or not isinstance(native, AssetBaseCfg) or spawn is None:
        raise NotImplementedError(f'Entity {path}: kinematic is bound for rigid objects and static props, '
                                  'not for articulations')
    if not isinstance(native, RigidObjectCfg):
        native = RigidObjectCfg(prim_path=native.prim_path, spawn=spawn, collision_group=native.collision_group)
        # The config class copies its fields: edit the copy it holds.
        spawn = native.spawn
    fragment = sim.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)
    props = spawn.rigid_props
    mapping = None if props is None else fragment_mapping(props, '')
    if props is None:
        # A bare fragment creates the rigid body on the spawn prim when the asset has none.
        spawn.rigid_props = fragment
    elif mapping is None:
        props = props.copy()
        props.kinematic_enabled = True
        spawn.rigid_props = props
    else:
        # Later fragments override earlier ones per attribute.
        mapping = {pattern: [*(fragments if isinstance(fragments, (list, tuple)) else [fragments]), fragment]
                   for pattern, fragments in mapping.items()}
        spawn.rigid_props = mapping if isinstance(props, dict) else mapping['']
    return native


@lru_cache(maxsize=1)
def _config_type():
    import torch
    from isaaclab.managers import CommandTerm, CommandTermCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_box_minus

    class KinematicPose(CommandTerm):
        """Writes the Task-set (else scene-motion) root pose of one kinematic entity, with velocity =
        displacement / step_dt.

        Environments reset since the last update keep the pose reset gave them (a path motion's start).
        Without a Task move or motion the entity stays put, with zero velocity."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.asset = env.scene[cfg.name]
            self.fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self.motion = None
            if cfg.motion is not None:
                from shared.assembly.motion import Motion
                from .python_task import scene_info
                self.motion = Motion(cfg.motion, cfg.base, cfg.areas, self.num_envs, self.device, package=cfg.package,
                                     scene=lambda: scene_info(env), name=cfg.name)

        @property
        def command(self):
            data = self.asset.data
            return torch.cat((data.root_link_pos_w.torch - self._env.scene.env_origins,
                              wxyz(data.root_link_quat_w.torch)), -1)

        def _resample_command(self, env_ids):
            self.fresh[env_ids] = True
            self.moving[env_ids] = False
            if self.motion is None:
                return
            self.motion.reset(env_ids)
            start = self.motion.start(env_ids)
            if start is not None:  # a path: the reset pose is on it
                rows = torch.arange(self.num_envs, device=self.device)[env_ids]
                pose = torch.cat((start[:, :3] + self._env.scene.env_origins[rows], xyzw(start[:, 3:7])), -1)
                self.asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=rows)
                self.asset.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(len(rows), 6, device=self.device),
                                                                 env_ids=rows)

        def _update_command(self):
            from .python_task import task_set
            value = task_set(self._env, self.cfg.name)
            if value is None and self.motion is not None:
                value = self.motion.step(self.command, self._env.step_dt, ~self.fresh)
            fresh, self.fresh[:] = self.fresh.clone(), False
            if value is None:
                # Stop: write zero velocity once where the entity was moving.
                rows = torch.nonzero(self.moving & ~fresh).flatten()
                if len(rows):
                    zero = torch.zeros(len(rows), 6, device=self.device)
                    self.asset.write_root_link_velocity_to_sim_index(root_velocity=zero, env_ids=rows)
                self.moving[:] = False
                return
            rows = torch.nonzero(~fresh).flatten()
            if not len(rows):
                return
            data, origins = self.asset.data, self._env.scene.env_origins[rows]
            position = data.root_link_pos_w.torch[rows]
            orientation = data.root_link_quat_w.torch[rows]
            value = value[rows].to(position.dtype)
            target = value[:, :3] + origins
            rotation = (xyzw(value[:, 3:7] / value[:, 3:7].norm(dim=-1, keepdim=True)) if value.shape[1] == 7
                        else orientation)
            dt = self._env.step_dt
            velocity = torch.cat(((target - position) / dt, quat_box_minus(rotation, orientation) / dt), -1)
            self.asset.write_root_link_pose_to_sim_index(root_pose=torch.cat((target, rotation), -1), env_ids=rows)
            self.asset.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=rows)
            self.moving[rows] = True

        def _update_metrics(self):
            pass

    @configclass
    class KinematicPoseCfg(CommandTermCfg):
        class_type: type = KinematicPose
        name: str = None
        motion: dict = None
        base: dict = None
        areas: dict = None
        package: str = None
    return KinematicPoseCfg


def attach_kinematic(commands, settings, package=None):
    """One command term per kinematic entity; its native scene key is the Task's ``set`` name."""
    from shared.assembly.abstract import reset_areas
    from shared.assembly.kinematic import kinematic_entities, kinematic_poses
    from shared.assembly.motion import entity_motions
    motions, poses, areas = entity_motions(settings), kinematic_poses(settings), reset_areas(settings)
    for key in kinematic_entities(settings):
        term = 'kinematic_' + key
        if term in commands:
            raise ValueError('Reserved kinematic command name: ' + term)
        # Resampled only on reset: the episode never reaches this interval.
        commands[term] = _config_type()(name=key, motion=motions.get(key), base=poses[key], areas=areas, package=package,
                                        resampling_time_range=(1e9, 1e9), debug_vis=False)
