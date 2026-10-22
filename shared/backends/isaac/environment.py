"""Configured Isaac Lab environment. The SDK owns simulation and episode steps."""

import math

from shared.integrations.rlinf.terminal import snapshot


from shared.backends.environment import Environment


def _ensure_native_horizon(config):
    from isaaclab.envs import mdp
    from isaaclab.managers import TerminationTermCfg

    if config.terminations is None:
        config.terminations = {}
    terms = config.terminations if isinstance(config.terminations, dict) else vars(config.terminations)
    if any(getattr(term, "func", None) is mdp.time_out and getattr(term, "time_out", False)
           for term in terms.values()):
        return
    if "workspace_time_limit" in terms:
        raise ValueError("workspace_time_limit is reserved for the training horizon")
    timeout = TerminationTermCfg(func=mdp.time_out, time_out=True)
    if isinstance(config.terminations, dict):
        config.terminations["workspace_time_limit"] = timeout
    else:
        config.terminations.workspace_time_limit = timeout


def build_config(prepared, *, parameters, num_envs, seed, horizon, viewer=None):
    """Compile the common scene/task descriptions into native SDK configuration."""
    from copy import deepcopy
    import isaaclab.sim as sim
    from isaaclab.envs import ManagerBasedRLEnvCfg
    from .scene import build_scene_config, event_configuration
    from .task_config import build_task_config

    settings = prepared.settings
    simulation = deepcopy(settings.get("simulation", {}))
    simulation["device"] = parameters.get("device", simulation.get("device", "cuda:0"))
    if "physx" in simulation:
        simulation["physx"] = sim.PhysxCfg(**simulation["physx"])
    if "physics_material" in simulation:
        simulation["physics_material"] = sim.RigidBodyMaterialCfg(**simulation["physics_material"])
    world = deepcopy(settings.get("world", {}))
    materials = world.pop("contact_materials", {})
    decimation = world.pop("decimation", 1)
    world["num_envs"] = num_envs
    config = ManagerBasedRLEnvCfg(
        sim=sim.SimulationCfg(**simulation), decimation=decimation, seed=seed,
        scene=build_scene_config(prepared.description, factories=prepared.factories, **world),
        **build_task_config(settings, term_registry=prepared.terms, task_presets=prepared.tasks),
    )
    # Scene assembly operations are regular SDK events after task reset events.
    from .contact_materials import events as material_events
    scene_events = {**material_events(materials), **event_configuration(prepared.description)}
    for name, term in scene_events.items():
        if name in config.events:
            raise ValueError(f"Scene event name is reserved: {name}")
        config.events[name] = term
    # Lab takes ceil(seconds / step_dt), so avoid a rounding-induced extra step.
    config.episode_length_s = math.nextafter(horizon * config.sim.dt * decimation, 0.0)
    _ensure_native_horizon(config)
    # Lab's velocity preset resets only its controlled robot. Every authored
    # physical asset must first share the selected terrain/environment origin;
    # task randomizers and assembly placement rules then run on that baseline.
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import EventTermCfg
    from .scene import scene_metadata, reset_authored_assets
    metadata = scene_metadata(prepared.description)
    asset_names = [key for path, key in metadata.keys.items()
                   if metadata.scopes[path] == "environment"
                   and isinstance(getattr(config.scene, key), AssetBaseCfg)]
    if {"assembly_defaults", "assembly_initial_poses"} & config.events.keys():
        raise ValueError("Scene event name is reserved: assembly_defaults/assembly_initial_poses")
    if asset_names:
        config.events = {"assembly_initial_poses": EventTermCfg(
            func=reset_authored_assets, mode="startup", params={"asset_names": asset_names}),
            "assembly_defaults": EventTermCfg(
            func=reset_authored_assets, mode="reset", params={"asset_names": asset_names}), **config.events}
    if viewer is not None:
        config.scene.clone_in_fabric = False
        config.viewer.origin_type, config.viewer.env_index = "env", 0
        for name, value in viewer.items():
            setattr(config.viewer, name, value)
    return config


def launch_application(options):
    import os
    if options["headless"]:
        os.environ.pop("DISPLAY", None)
    elif not os.environ.get("DISPLAY"):
        raise RuntimeError("GUI requires DISPLAY; use a desktop terminal or record")
    from isaaclab.app import AppLauncher
    return AppLauncher(**options).app


def make_environment(parameters, num_envs, seed, horizon, *, scene_config, project_package=None,
                     factories=None, terms=None, tasks=None,
                     headless=None, enable_cameras=None, viewer=None):
    from shared.assembly.project import prepare_scene
    from shared.isaac_state import ProcessExit

    prepared = prepare_scene(parameters, scene_config=scene_config, project_package=project_package,
                             factories=factories, terms=terms, tasks=tasks,
                             headless=headless, enable_cameras=enable_cameras)
    application = launch_application(prepared.application)
    native = None
    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from shared.integrations.rlinf.terminal import TerminalObservationMixin, validate_terminal_observations

        class NativeEnv(TerminalObservationMixin, ManagerBasedRLEnv):
            pass

        config = build_config(prepared, parameters=parameters, num_envs=num_envs,
                              seed=seed, horizon=horizon, viewer=viewer)
        native = NativeEnv(cfg=config, render_mode="rgb_array" if prepared.application["enable_cameras"] else None)
        if native.max_episode_length != horizon:
            raise ValueError(f"Native horizon {native.max_episode_length} does not match requested {horizon}")
        validate_terminal_observations(native.observation_manager)
        return Environment(native), ProcessExit(application)
    except BaseException:
        # Kit teardown can block after partial construction on this installation.
        # Reuse the existing worker exit owner, preserving the original traceback.
        ProcessExit(application).close()
        raise
