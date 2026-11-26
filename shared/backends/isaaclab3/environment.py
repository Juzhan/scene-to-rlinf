"""Common project configuration bound to Isaac Lab 3's native ManagerBasedRLEnv (Newton or OvPhysX).

Isaac Lab 3 owns physics, managers and episode steps; its ``launch_simulation`` decides
whether a Kit application is needed (kitless for Newton and OvPhysX without Kit cameras/visualizers).
"""
import math
from types import SimpleNamespace
from shared.backends.environment import Environment


ENGINES = ('newton', 'ovphysx')
# Renderers for the preview visualizer and scene cameras: Newton (GL preview, Warp cameras) or OVRTX.
RENDERERS = ('newton', 'ovrtx')


def renderer_name(parameters):
    renderer = parameters.get('renderer', 'newton')
    if renderer not in RENDERERS:
        raise ValueError(f'isaaclab3 renderer must be one of {RENDERERS}, got {renderer!r}')
    return renderer


def physics_config(simulation, engine='newton'):
    """Isaac Lab 3's own locomotion physics preset for the selected engine, with scene overrides.

    ``newton`` (default): MuJoCo-Warp with contact capacity sized from the model (contacts.py);
    ``simulation.newton`` sets NewtonCfg fields and its ``solver`` mapping MJWarpSolverCfg fields. ``ovphysx``: kitless PhysX; ``simulation.physx``
    sets the OvPhysxCfg fields (GPU buffers), as it sets PhysxCfg on the isaac backend. Each
    engine ignores the other's block, as mjlab ignores ``physx``."""
    from copy import deepcopy
    from isaaclab_tasks.core.velocity.velocity_env_cfg import RoughPhysicsCfg
    if engine not in ENGINES:
        raise ValueError(f'isaaclab3 physics must be one of {ENGINES}, got {engine!r}')
    if engine == 'ovphysx':
        physics = deepcopy(RoughPhysicsCfg().ovphysx)
        targets = ((physics, dict(simulation.get('physx', {}))),)
    else:
        from .contacts import NewtonSizedMJWarpManager
        physics = deepcopy(RoughPhysicsCfg().newton_mjwarp)
        physics.class_type = physics.solver_cfg.class_type = NewtonSizedMJWarpManager
        physics.solver_cfg.nconmax = physics.solver_cfg.njmax = None
        overrides = dict(simulation.get('newton', {}))
        targets = ((physics.solver_cfg, overrides.pop('solver', {})), (physics, overrides))
    for owner, values in targets:
        for name, value in values.items():
            if name in vars(owner):
                setattr(owner, name, value)
            elif engine == 'ovphysx':
                # Other PhysxCfg scene options are USD PhysicsScene attributes OvPhysX does not author.
                raise NotImplementedError(f'simulation.physx: OvPhysxCfg does not expose {name!r}')
            else:
                raise ValueError(f'simulation.newton: unknown {type(owner).__name__} field {name!r}')
    return physics


PREVIEW_CAMERA = 'rlinf_preview_camera'  # scene key of the camera that renders the preview viewpoint (render: camera)


def preview_camera_config(width, height):
    """A camera at the preview viewpoint, rendered like the scene cameras (Newton Warp, or OVRTX);
    Preview.camera_frame gives it the viewer camera's pose and field of view before each frame."""
    from .assets import camera
    config = camera(width=width, height=height, data_types=('rgb',), clipping_range=(0.02, 200.0))
    config.prim_path = '{ENV_REGEX_NS}/RLinfPreviewCamera'
    return config


def build_config(prepared, *, parameters, num_envs, seed, horizon, visualize=False, preview_camera=None):
    from isaaclab.envs import ManagerBasedRLEnvCfg, mdp
    from isaaclab.managers import EventTermCfg, ObservationGroupCfg, TerminationTermCfg
    from isaaclab.sim import SimulationCfg
    from .abstract_terms import attach_abstract
    from .assets import use_renderer
    from .kinematic import attach_kinematic
    from .python_task import attach_task
    from .scene import authored_reset_assets, build_scene_config, event_configuration, reset_authored_assets
    settings = prepared.settings
    simulation = settings.get('simulation', {})
    sim = SimulationCfg(device=parameters.get('device', 'cuda:0'), dt=simulation.get('dt', .005),
                        gravity=tuple(simulation.get('gravity', (0., 0., -9.81))),
                        physics=physics_config(simulation, parameters.get('physics', 'newton')))
    # Visualizers redraw at the render cadence; by default once per policy step.
    sim.render_interval = simulation.get('render_interval', settings['world']['decimation'])
    if 'physics_material' in simulation:
        # As in Isaac Lab 3's own tasks: PhysX reads the combine modes, Newton the shared fields.
        from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
        sim.physics_material = PhysxRigidBodyMaterialCfg(**simulation['physics_material'])
    scene = build_scene_config(prepared.description, factories=prepared.factories, num_envs=num_envs,
                               env_spacing=settings.get('world', {}).get('env_spacing', 4.0))
    if preview_camera is not None:
        setattr(scene, PREVIEW_CAMERA, preview_camera_config(*preview_camera))
    use_renderer(scene, renderer_name(parameters))
    # Explicit actuators run on Isaac Lab's Python models, the explicit torque the isaac and mjlab
    # backends compute. Newton's native path (Isaac Lab 3's default) runs DC motors as an implicit
    # PD, which trained legged policies do not survive, and rejects actuator classes it lacks.
    sim.use_newton_actuators = False

    def group(source):
        observation = ObservationGroupCfg(enable_corruption=False, concatenate_terms=True)
        observation.value = prepared.terms['observations'][source]()
        return observation
    # Isaac Lab's default event manager resets every asset to its initial state; the authored
    # defaults below do the same for scene assets, so the whole ordered set is ours.
    events = {}
    asset_names = authored_reset_assets(scene, prepared.description)
    if asset_names:
        events['assembly_initial_poses'] = EventTermCfg(func=reset_authored_assets, mode='startup',
                                                        params={'asset_names': asset_names})
        events['assembly_defaults'] = EventTermCfg(func=reset_authored_assets, mode='reset',
                                                   params={'asset_names': asset_names})
    from .contact_materials import events as material_events
    from .randomization import events as randomization_events
    scene_events = {**material_events(settings.get('world', {}).get('contact_materials', {})),
                    **randomization_events(settings.get('_randomization', ()), settings,
                                           parameters.get('physics', 'newton')),
                    **event_configuration(prepared.description)}
    for name, term in scene_events.items():
        if name in events:
            raise ValueError(f'Scene event name is reserved: {name}')
        events[name] = term
    output = {'rewards': {}, 'terminations': {}, 'events': events}
    attach_task(output, prepared.terms['object_api'], settings)
    output['terminations']['workspace_time_limit'] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    commands = {}
    package = prepared.terms['object_api']['module']  # custom scene motion lives in the project
    attach_abstract(commands, settings, visualize=visualize, package=package)
    attach_kinematic(commands, settings, package=package)
    decimation = settings['world']['decimation']
    # Managers set attributes such as debug_vis on their cfg, so terms go in attribute containers.
    terms = lambda entries: SimpleNamespace(**entries)
    return ManagerBasedRLEnvCfg(
        sim=sim, scene=scene, seed=seed, decimation=decimation,
        # Lab takes ceil(seconds / step_dt), so avoid a rounding-induced extra step.
        episode_length_s=math.nextafter(horizon * sim.dt * decimation, 0.),
        actions=terms({'control_chain': prepared.terms['actions']['control_chain']()}),
        observations=terms({source: group(source) for source in settings['observations']['policy']}),
        events=terms(events), rewards=terms(output['rewards']), terminations=terms(output['terminations']),
        commands=terms(commands) if commands else None)


class Owner:
    """Closes the runtimes ``launch_simulation`` started (none when kitless)."""
    application = None  # No Kit application: the Newton visualizer owns any preview window.

    def __init__(self, launch):
        self.launch = launch

    def close(self):
        if self.launch is not None:
            launch, self.launch = self.launch, None
            launch.__exit__(None, None, None)


def camera_origin(native, viewer):
    """Frame of the preview eye/lookat: the followed robot's root in environment 0, else its origin."""
    if viewer.get('follow') is not None:
        return native.scene[viewer['follow']].data.root_link_pos_w.torch[0].cpu().numpy()
    return native.scene.env_origins[0].cpu().numpy()


class Preview:
    """Isaac Lab 3's Newton visualizer showing environment 0 from the preview eye/lookat.

    The ``newton`` renderer rasterizes with GL, ``ovrtx`` path-traces with OVRTX; both share the
    viewer camera and frame read-back. ``window`` opens its interactive window (view); otherwise
    it renders offscreen and frames come from its own ``render_rgb_array`` (record). The
    environment's render step redraws the window; this only aims the camera and reports closing."""
    def __init__(self, viewer, *, window, renderer='newton'):
        self.viewer, self.window, self.renderer = viewer, window, renderer
        self.origin = None
        # render: camera draws the viewpoint with the scene cameras' renderer (preview_camera_config).
        self.camera = viewer.get('render') == 'camera'

    def config(self):
        from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
        width, height = self.viewer.get('resolution', (1280, 720))
        config = NewtonRTXVisualizerCfg if self.renderer == 'ovrtx' else NewtonGLVisualizerCfg
        return config(headless=not self.window, window_width=width, window_height=height,
                      visible_env_indices=[0], enable_live_plots=False)

    def visualizer(self, native):
        kind = 'newton_rtx' if self.renderer == 'ovrtx' else 'newton_gl'
        return next(item for item in native.sim.visualizers if item.cfg.visualizer_type == kind)

    def aim(self, native):
        import numpy as np
        if self.origin is not None and self.viewer.get('follow') is None:
            return
        origin = camera_origin(native, self.viewer)
        visualizer = self.visualizer(native)
        if self.origin is not None and self.window:
            # Carry the window camera with the robot; a user's orbit and zoom stay as they are.
            # The Newton visualizers expose no follow mode or camera handle of their own.
            visualizer._viewer.camera.translate(origin - self.origin)
        else:
            eye = origin + np.asarray(self.viewer.get('eye', (7.5, 7.5, 7.5)), dtype=float)
            lookat = origin + np.asarray(self.viewer.get('lookat', (0., 0., 0.)), dtype=float)
            visualizer.set_camera_view(tuple(eye), tuple(lookat))
        self.origin = origin

    def camera_frame(self, native):
        """The viewpoint as the scene cameras' renderer draws it: the preview camera moved to the viewer
        camera (which the window's mouse and the follow mode move), environment 0, uint8 H x W x 3."""
        import torch
        viewer = self.visualizer(native)._viewer.camera
        sensor = native.scene.sensors[PREVIEW_CAMERA]
        device = native.device
        eye = torch.tensor([[float(v) for v in viewer.pos]], device=device)
        # Along the direction the viewer draws (yaw/pitch): its pivot is only the orbit centre, which the
        # RTX viewer leaves in place when the view is set from an eye and a target.
        front = viewer.get_front()
        target = eye + torch.tensor([[float(front[i]) for i in range(3)]], device=device)
        env = torch.tensor([0], device=device)
        view = (tuple(eye[0].tolist()), tuple(target[0].tolist()), float(viewer.fov))
        # Every frame: OVRTX takes a camera pose only when it is set, and drops one set before its first render.
        sensor.set_world_poses_from_view(eye, target, env_ids=env)
        if getattr(self, 'view', None) != view:
            if getattr(self, 'view', (None,) * 3)[2] != view[2]:  # the visualizer derives it from its focal length
                height, width = sensor.data.output['rgb'].torch.shape[1:3]
                focal = height / 2 / math.tan(math.radians(view[2]) / 2)  # pixels, square
                matrix = torch.tensor([[[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]]], device=device)
                # The spawned lens's 24 mm focal length; the aperture follows (else 1 mm wide).
                sensor.set_intrinsic_matrices(matrix, focal_length=24.0, env_ids=env)
            sensor.update(0.0, force_recompute=True)  # otherwise the image is from before the move
            self.view = view
        image = sensor.data.output['rgb'].torch[0, ..., :3]
        return (image if image.dtype == torch.uint8 else (image.clamp(0, 1) * 255).to(torch.uint8)).cpu().numpy()

    def frame(self, native):
        if self.camera:
            return self.camera_frame(native)
        # As Lab's own VideoRecorder: without continuous rendering, nothing has dispatched the
        # marker callbacks (abstract entities) since the last step, so do it before capturing.
        visualizer = self.visualizer(native)
        if not native.sim.is_rendering:
            native.sim.forward()
            if visualizer.supports_markers():
                native.sim.vis_marker_registry.dispatch_callbacks()
        return visualizer.render_rgb_array()

    def closed(self, native):
        return not self.visualizer(native).is_running()


def make_environment(parameters, num_envs, seed, horizon, *, scene_config, project_package=None,
                     headless=None, enable_cameras=None, viewer=None, **registries):
    from shared.assembly.framing import complete_viewer
    from shared.assembly.project import prepare_scene
    if viewer is not None:
        # Preview asks Isaac to enable camera rendering for its viewport video. The Newton
        # visualizer draws the window/video itself, so scene cameras keep the scene's own setting,
        # unless the preview shows one of them (view/record --camera: viewer['scene_cameras']).
        viewer = dict(viewer)
        enable_cameras = True if viewer.pop('scene_cameras', False) or viewer.get('render') == 'camera' else None
    prepared = prepare_scene(parameters, scene_config=scene_config, project_package=project_package,
                             headless=True, enable_cameras=enable_cameras, **registries)
    viewer = complete_viewer(viewer, prepared.description)
    preview = (Preview(viewer or {}, window=headless is False, renderer=renderer_name(parameters))
               if viewer is not None or headless is False else None)
    cfg = build_config(prepared, parameters=parameters, num_envs=num_envs, seed=seed, horizon=horizon,
                       visualize=preview is not None,
                       preview_camera=(viewer or {}).get('resolution', (1280, 720)) if preview and preview.camera else None)
    if preview is not None:
        cfg.sim.visualizer_cfgs = [preview.config()]
        # Newton infers visual-geometry import from CLI visualizers or continuous rendering; a
        # configured offscreen visualizer satisfies neither, so request it explicitly. OvPhysX
        # has no such switch.
        if getattr(cfg.sim.physics, 'load_visual_shapes', False) is None:
            cfg.sim.physics.load_visual_shapes = True
    from isaaclab.app import launch_simulation
    launch = launch_simulation(cfg, {'headless': True})
    launch.__enter__()
    owner, native = Owner(launch), None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from shared.backends.terminal import TerminalObservationMixin, validate_terminal_observations
        from .scene import newton_connections, physx_connections, refresh_newton_frames

        class NativeEnv(TerminalObservationMixin, ManagerBasedRLEnv):
            backend_name = 'isaaclab3'
            physics_name = parameters.get('physics', 'newton')

            def _init_sim(self):
                with physx_connections(self, prepared.description):
                    super()._init_sim()

            def _reset_idx(self, env_ids):
                # Lab 3 passes slice(None) for all environments; shared resets take index tensors.
                if isinstance(env_ids, slice):
                    env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
                result = super()._reset_idx(env_ids)
                # Observations follow the reset; Newton frame sensors would still hold pre-reset frames.
                refresh_newton_frames(self)
                return result

            def render(self, recompute=False):
                if preview is None or preview.window:
                    raise ValueError('Create the isaaclab3 environment with a recording viewer to render video frames')
                return preview.frame(self)

            def show_preview(self):
                """view with render: camera: fill the window with the viewpoint as the camera renderer draws it."""
                if preview is not None and preview.window and preview.camera:
                    self.show_image(preview.camera_frame(self))

            def camera_image(self, name):
                """RGB (uint8 H x W x 3) of scene camera ``name`` in environment 0, from the scene's camera
                renderer (Newton Warp, or OVRTX): what the policy reads, unlike the preview window's own view."""
                sensor = self.scene.sensors.get(name)
                output = getattr(getattr(sensor, 'data', None), 'output', None)
                if output is None or 'rgb' not in output:
                    cameras = [key for key, item in self.scene.sensors.items()
                               if 'rgb' in (getattr(getattr(item, 'data', None), 'output', None) or {})]
                    raise ValueError(f'No rgb camera {name!r} in the scene; rgb cameras: {cameras or "none"}')
                image = output['rgb'].torch[0, ..., :3]
                return (image if image.dtype == torch.uint8 else (image.clamp(0, 1) * 255).to(torch.uint8)).cpu().numpy()

            def show_image(self, image):
                """Fill the preview window with ``image`` this frame (view --camera)."""
                if preview is None or not preview.window:
                    raise ValueError('Create the isaaclab3 environment with a preview window to show images')
                preview.visualizer(self)._viewer.log_image('camera', image, fullscreen=True)

            @property
            def window_closed(self):
                return preview is not None and preview.window and preview.closed(self)

            def reset(self, *args, **kwargs):
                result = super().reset(*args, **kwargs)
                if preview is not None:
                    preview.aim(self)
                return result

            def step(self, action):
                result = super().step(action)
                if preview is not None:
                    preview.aim(self)
                return result

        with newton_connections(prepared.description, cfg.sim.physics, cfg.sim.dt):
            native = NativeEnv(cfg=cfg)
        if native.max_episode_length != horizon:
            raise ValueError(f'Native horizon {native.max_episode_length} does not match requested {horizon}')
        validate_terminal_observations(native.observation_manager)
        return Environment(native), owner
    except BaseException:
        if native is not None:
            native.close()
        owner.close()
        raise
