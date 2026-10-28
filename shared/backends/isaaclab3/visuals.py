"""Editor geometry snapshot on Isaac Lab 3: kitless Newton, live body link poses.

``tools/export_scene_visuals.py`` owns the backend-neutral USD mesh extraction and cache;
this module only builds the native scene and reports poses (shared wxyz) in it. Geometry
does not depend on the physics engine, so Newton is used whichever one training selects.
"""
from contextlib import contextmanager

# Files whose changes invalidate this backend's exported geometry (workspace-relative globs).
SOURCES = ('shared/backends/isaaclab3/assets.py', 'shared/backends/isaaclab3/scene.py',
           'shared/backends/isaaclab3/spawners.py', 'shared/backends/isaaclab3/visuals.py',
           'shared/backends/isaac/initial_state.py')


def launch_options(prepared):
    return {}


class _Kitless:
    def close(self):
        pass


def launch(prepared):
    """No application to start: Newton builds the scene from the in-memory USD stage."""
    return _Kitless()


def _pose(tensor):
    from .frames import wxyz
    return tensor[:3].tolist(), wxyz(tensor[3:7]).tolist()


class Snapshot:
    def __init__(self, scene, prepared):
        from .scene import scene_metadata
        self.scene, self.prepared = scene, prepared
        self.metadata = scene_metadata(prepared.description)
        self.stage = scene.stage
        self.origin = scene.env_origins[0].cpu().tolist()

    def entity(self, path, node):
        """Prim, default/initial state and live link poses of one entity in environment 0."""
        from .frames import wxyz
        key = self.metadata.keys[path]
        result = {'prim_path': self.metadata.prim_paths[path].replace('{ENV_REGEX_NS}', '/World/envs/env_0')}
        initial = getattr(self.prepared.factories[node.factory](**node.parameters), 'init_state', None)
        if initial is not None:
            result['default_pose'] = {'position': list(initial.pos), 'orientation': list(wxyz(tuple(initial.rot)))}
        if key in self.scene.articulations or key in self.scene.rigid_objects:
            # A rigid object is one body: sensors mount on it by the same link names.
            asset = self.scene[key]
            data = asset.data
            velocity = data.default_root_vel.torch[0].cpu()
            result['initial_state'] = {'basis': 'Native asset defaults before task reset randomization',
                                       'linear_velocity': velocity[:3].tolist(),
                                       'angular_velocity': velocity[3:].tolist(),
                                       'body_names': list(asset.body_names), 'root_body': asset.body_names[0]}
        if key in self.scene.articulations:
            result['initial_state'].update(
                joint_positions=dict(zip(asset.joint_names, data.default_joint_pos.torch[0].cpu().tolist())),
                joint_velocities=dict(zip(asset.joint_names, data.default_joint_vel.torch[0].cpu().tolist())))
            links = data.body_link_pose_w.torch[0].cpu()
            result['links'] = {name: _pose(links[index]) for index, name in enumerate(asset.body_names)}
            # Root from the same link tensor: for a fixed-base articulation Newton reports the
            # written root pose while its links stay where they were spawned.
            result['root_pose'] = result['links'][asset.body_names[0]]
        return result


@contextmanager
def snapshot(prepared):
    """The geometry-only scene of ``prepared`` on a fresh stage, at its authored defaults."""
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.utils.seed import configure_seed
    from .environment import physics_config
    from .scene import build_scene_config
    simulation = None
    try:
        configure_seed(42)
        scene_cfg = build_scene_config(prepared.description, factories=prepared.factories, num_envs=1, env_spacing=4)
        sim_utils.create_new_stage()
        # Explicit actuators as in environment.py: Newton's native path rejects classes it lacks.
        simulation = SimulationContext(SimulationCfg(device='cuda:0' if torch.cuda.is_available() else 'cpu',
                                                     physics=physics_config({}, 'newton'), use_newton_actuators=False))
        scene = InteractiveScene(scene_cfg)
        if not scene.articulations and not scene.rigid_objects:
            # Only static geometry (an incremental export of a table or terrain tile): the stage
            # holds it already, and Newton's MuJoCo solver rejects a model without any joint.
            yield Snapshot(scene, prepared)
            return
        simulation.reset()
        for key in [*scene.articulations, *scene.rigid_objects]:
            asset = scene[key]
            pose = asset.data.default_root_pose.torch[:1].clone()
            pose[:, :3] += scene.env_origins[:1]
            asset.write_root_pose_to_sim_index(root_pose=pose)
            if key in scene.articulations:
                asset.write_joint_state_to_sim_index(position=asset.data.default_joint_pos.torch[:1].clone(),
                                                     velocity=asset.data.default_joint_vel.torch[:1].clone())
        simulation.forward()  # forward kinematics of the written state, without a physics step
        yield Snapshot(scene, prepared)
    finally:
        if simulation is not None:
            # As Isaac Lab 3's build_simulation_context: a later snapshot starts a new context.
            simulation.stop()
            SimulationContext.clear_instance()
