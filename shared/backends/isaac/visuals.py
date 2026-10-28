"""Editor geometry snapshot on Isaac Lab 2: Kit on CPU, PhysX link transforms.

``tools/export_scene_visuals.py`` owns the backend-neutral USD mesh extraction and cache;
this module only builds the native scene and reports poses (wxyz) in it.
"""
from contextlib import contextmanager
import os

# Files whose changes invalidate this backend's exported geometry (workspace-relative globs).
SOURCES = ('shared/backends/isaac/assets.py', 'shared/backends/isaac/scene.py',
           'shared/backends/isaac/environment.py', 'shared/backends/isaac/visuals.py',
           'shared/backends/isaac/robots/*.py')


def launch_options(prepared):
    """A running exporter serves another scene only with the same Kit options."""
    return dict(prepared.application)


def launch(prepared):
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from shared.backends.isaac.environment import launch_application
    from shared.isaac_state import ProcessExit
    # Stopping a standalone Kit timeline can block headless shutdown; ProcessExit flushes and exits.
    return ProcessExit(launch_application(prepared.application))


class Snapshot:
    def __init__(self, native, scene_cfg, prepared):
        from shared.backends.isaac.scene import scene_metadata
        self.native, self.scene_cfg, self.prepared = native, scene_cfg, prepared
        self.metadata = scene_metadata(prepared.description)
        self.stage = native.scene.stage
        self.origin = native.scene.env_origins[0].cpu().tolist()

    def entity(self, path, node):
        """Prim, default/initial state and live link poses of one entity in environment 0."""
        key = self.metadata.keys[path]
        scene = self.native.scene
        result = {'prim_path': self.metadata.prim_paths[path].replace('{ENV_REGEX_NS}', '/World/envs/env_0')}
        initial = getattr(self.prepared.factories[node.factory](**node.parameters), 'init_state', None)
        if initial is not None:
            result['default_pose'] = {'position': list(initial.pos), 'orientation': list(initial.rot)}
        if key in scene.articulations or key in scene.rigid_objects:
            asset = scene[key]
            result['initial_state'] = {
                'basis': 'Native asset defaults before task reset randomization',
                'linear_velocity': asset.data.default_root_state[0, 7:10].cpu().tolist(),
                'angular_velocity': asset.data.default_root_state[0, 10:13].cpu().tolist(),
            }
        if key in scene.rigid_objects:  # One body: sensors mount on it by the same link names.
            asset = scene.rigid_objects[key]
            result['initial_state'].update(body_names=list(asset.body_names), root_body=asset.body_names[0])
        if key in scene.articulations:
            asset = scene.articulations[key]
            result['initial_state'].update(
                body_names=list(asset.body_names), root_body=asset.body_names[0],
                joint_positions=dict(zip(asset.joint_names, asset.data.default_joint_pos[0].cpu().tolist())),
                joint_velocities=dict(zip(asset.joint_names, asset.data.default_joint_vel[0].cpu().tolist())))
            result['root_pose'] = (asset.data.root_pos_w[0].tolist(), asset.data.root_quat_w[0].tolist())
            # USD may retain the source asset's rest pose after a tensor joint reset; PhysX FK does not.
            links = asset.root_physx_view.get_link_transforms()[0].cpu()
            result['links'] = {name: (links[index, :3].tolist(), links[index, [6, 3, 4, 5]].tolist())
                               for index, name in enumerate(asset.body_names)}
        return result


@contextmanager
def snapshot(prepared):
    """The geometry-only scene of ``prepared`` on a fresh stage inside the running Kit."""
    from isaaclab.assets import AssetBaseCfg, ArticulationCfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.sim.utils.stage import attach_stage_to_usd_context, create_new_stage, use_stage
    from isaaclab.utils.seed import configure_seed
    from types import SimpleNamespace
    import torch
    from shared.backends.isaac.scene import build_scene_config, reset_authored_assets, scene_metadata
    simulation = scene_instance = None
    try:
        configure_seed(42)
        scene_cfg = build_scene_config(prepared.description, factories=prepared.factories,
                                       num_envs=1, env_spacing=4, replicate_physics=False,
                                       clone_in_fabric=False)
        # Pin articulation roots for this snapshot: disabling a USD world joint without the
        # assembly's root relocation can leave PhysX no valid root. Preview configs only.
        for config in vars(scene_cfg).values():
            if isinstance(config, ArticulationCfg):
                props = getattr(config.spawn, 'articulation_props', None)
                if props is not None:
                    props.fix_root_link = True
        # A long-lived exporter serves several snapshots; each starts from an empty stage.
        create_new_stage()
        simulation = SimulationContext(SimulationCfg(device='cpu', use_fabric=False))
        with use_stage(simulation.get_initial_stage()):
            scene_instance = InteractiveScene(scene_cfg)
            attach_stage_to_usd_context()
        native = SimpleNamespace(scene=scene_instance, sim=simulation, num_envs=1, device='cpu',
                                 cfg=SimpleNamespace(scene=scene_cfg))
        with use_stage(simulation.get_initial_stage()):
            simulation.reset()
        metadata = scene_metadata(prepared.description)
        names = [key for path, key in metadata.keys.items() if metadata.scopes[path] == 'environment'
                 and isinstance(getattr(scene_cfg, key), AssetBaseCfg)]
        reset_authored_assets(native, None, asset_names=[key for key in names if hasattr(native.scene[key], 'data')])
        for key in names:
            if not hasattr(native.scene[key], 'data'):
                # One environment without Fabric: the static view is complete and writes USD,
                # so the snapshot shares the environment origin like physical assets.
                initial = getattr(scene_cfg, key).init_state
                native.scene[key].set_world_poses(
                    positions=torch.tensor([initial.pos]) + native.scene.env_origins[:1],
                    orientations=torch.tensor([initial.rot], dtype=torch.float32))
        native.scene.write_data_to_sim()
        native.sim.physics_sim_view.update_articulations_kinematic()
        native.sim.forward()
        yield Snapshot(native, scene_cfg, prepared)
    finally:
        if scene_instance is not None:
            for group in ('articulations', 'rigid_objects', 'rigid_object_collections', 'deformable_objects', 'extras'):
                for asset in getattr(scene_instance, group, {}).values():
                    if hasattr(asset, '_clear_callbacks'):
                        asset._clear_callbacks()
        if simulation is not None:
            simulation.clear_all_callbacks()
            SimulationContext.clear_instance()
