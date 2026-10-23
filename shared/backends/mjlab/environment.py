"""Common project configuration bound to mjlab's native vector environment."""
import math
from shared.backends.environment import Environment
from shared.integrations.rlinf.terminal import snapshot
from shared.backends.mjlab.scene import HIDDEN_GEOM_GROUP

# Constraint rows per world. mujoco_warp's guess (64 for a small arm) overflows once two
# finger pads grip a box (about 115 rows), and it then drops rows silently: fingers pass
# their joint limits and the object is thrown. mjlab's own lift-cube task allocates 600.
NJMAX = 600


class TerminalObservationMixin:
    """Capture pre-reset state without replacing mjlab's step loop."""
    def step(self, actions):
        self._terminal_observation = None
        self._capturing_terminal = True
        try:
            obs, reward, terminated, truncated, extras = super().step(actions)
            info = snapshot(extras)
            info['autoreset'] = True
            if self._terminal_observation is not None:
                info['terminal_observation'] = self._terminal_observation
            return obs, reward, terminated, truncated, info
        finally:
            self._capturing_terminal = False

    def _reset_idx(self, env_ids=None):
        if getattr(self, '_capturing_terminal', False) and len(env_ids):
            # Native mjlab normally forwards after reset; refresh terminal
            # derived poses first so truncation bootstrap sees the final state.
            self.sim.forward()
            self.sim.sense()
            chain = self._assembly_control_chain
            state = chain.snapshot_state()
            try:
                # compute(False) returns the previous decision's cache in
                # mjlab 1.1.1. Read groups explicitly; our generated terms have
                # no native noise/delay/history or stateful modifiers.
                self._terminal_observation = snapshot({
                    name: self.observation_manager.compute_group(name, update_history=False)
                    for name in self.observation_manager.active_terms})
            finally:
                chain.restore_state(state)
        return super()._reset_idx(env_ids)


def build_config(prepared, *, num_envs, seed, horizon):
    from mjlab.envs import ManagerBasedRlEnvCfg
    from mjlab.envs import mdp
    from mjlab.managers import ObservationGroupCfg, TerminationTermCfg
    from mjlab.sim import SimulationCfg, MujocoCfg
    from .scene import build_scene_config, event_configuration
    from .python_task import attach_task
    settings = prepared.settings
    layout = {k: v for k, v in settings.items() if k in {
        'schema_version','name','entities','connections','simulation','authoring'}}
    layout['world'] = {k:v for k,v in settings.get('world',{}).items() if k != 'decimation'}
    scene, keys = build_scene_config(prepared.description, layout, prepared.factories)
    scene.num_envs = num_envs
    dt = settings.get('simulation', {}).get('dt', .005)
    decimation = settings['world']['decimation']
    cfg = ManagerBasedRlEnvCfg(scene=scene, seed=seed, decimation=decimation,
        episode_length_s=math.nextafter(horizon * dt * decimation, 0.),
        sim=SimulationCfg(njmax=NJMAX, mujoco=MujocoCfg(timestep=dt,
            gravity=tuple(settings.get('simulation', {}).get('gravity', [0.,0.,-9.81])))),
        actions={'control_chain': prepared.terms['actions']['control_chain']()},
        observations={source: ObservationGroupCfg(terms={'value':prepared.terms['observations'][source]()})
                      for source in settings['observations']['policy']})
    # Authored root resets run after mjlab's default reset to the initial state.
    cfg.events.update(event_configuration(prepared.description, keys))
    output = {'rewards':cfg.rewards, 'terminations':cfg.terminations, 'events':cfg.events}
    attach_task(output, prepared.terms['object_api'], settings)
    cfg.terminations['workspace_time_limit'] = TerminationTermCfg(func=mdp.time_out, time_out=True)
    return cfg


class Owner:
    """mjlab needs no external application; native.close owns resources."""
    application = None
    def close(self): pass



def aim_camera(camera, native, viewer):
    """Free camera at the preview eye/lookat, relative to environment 0."""
    import mujoco
    import numpy as np
    origin = native.scene.env_origins[0].cpu().numpy()
    eye = origin + np.asarray(viewer.get('eye', (7.5, 7.5, 7.5)), dtype=float)
    lookat = origin + np.asarray(viewer.get('lookat', (0., 0., 0.)), dtype=float)
    direction = lookat - eye
    distance = float(np.linalg.norm(direction))
    if not distance:
        raise ValueError('Preview camera eye and lookat coincide')
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = math.degrees(math.atan2(direction[1], direction[0]))
    camera.elevation = math.degrees(math.asin(direction[2] / distance))


def copy_environment_zero(native, data):
    """Environment 0's GPU state into a CPU MjData of the same single-world model."""
    import mujoco
    model, state = native.sim.mj_model, native.sim.data
    data.qpos[:] = state.qpos[0].cpu().numpy()
    data.qvel[:] = state.qvel[0].cpu().numpy()
    if model.nmocap:
        data.mocap_pos[:] = state.mocap_pos[0].cpu().numpy()
        data.mocap_quat[:] = state.mocap_quat[0].cpu().numpy()
    mujoco.mj_forward(model, data)


class Window:
    """MuJoCo's own viewer showing environment 0 while the shared preview loop runs the policy.

    The window is passive: it neither steps physics nor calls the policy."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.handle = None
        self.closed = False

    def show(self, native):
        """Refresh the window after reset/step; marks it closed once the user has closed it."""
        import mujoco
        import mujoco.viewer
        if self.handle is None:
            model = native.sim.mj_model
            self.data = mujoco.MjData(model)
            copy_environment_zero(native, self.data)
            self.handle = mujoco.viewer.launch_passive(model, self.data)
            with self.handle.lock():
                aim_camera(self.handle.cam, native, self.viewer)
                self.handle.opt.geomgroup[HIDDEN_GEOM_GROUP] = 1
        if not self.handle.is_running():
            self.closed = True
            return
        with self.handle.lock():
            copy_environment_zero(native, self.data)
        self.handle.sync()

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None


class Video:
    """MuJoCo image of environment 0 alone, seen from the preview eye/lookat.

    The coordinates are relative to environment 0 as in the Isaac preview; mjlab's
    own rgb_array renderer would overlay every environment world. The OpenGL backend is
    $MUJOCO_GL, else EGL: mujoco fixes its Renderer backend at import, before this runs."""
    def __init__(self, viewer):
        self.viewer = viewer
        self.renderer = None

    def render(self, native):
        from importlib import import_module
        import os
        import mujoco
        import numpy as np
        model = native.sim.mj_model
        if self.renderer is None:
            width, height = self.viewer.get('resolution', (1280, 720))
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
            self.context = import_module(f"mujoco.{os.environ.get('MUJOCO_GL') or 'egl'}").GLContext(width, height)
            self.context.make_current()
            self.renderer = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.renderer)
            self.viewport = mujoco.MjrRect(0, 0, width, height)
            self.scene = mujoco.MjvScene(model, maxgeom=10000)
            self.option = mujoco.MjvOption()
            self.option.geomgroup[HIDDEN_GEOM_GROUP] = 1
            self.pixels = np.empty((height, width, 3), dtype=np.uint8)
            self.data = mujoco.MjData(model)
            self.camera = mujoco.MjvCamera()
            aim_camera(self.camera, native, self.viewer)
        copy_environment_zero(native, self.data)
        self.context.make_current()
        mujoco.mjv_updateScene(model, self.data, self.option, None, self.camera,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(self.viewport, self.scene, self.renderer)
        mujoco.mjr_readPixels(self.pixels, None, self.viewport, self.renderer)
        return np.flipud(self.pixels).copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.free()
            self.context.free()
            self.renderer = None


def make_environment(parameters, num_envs, seed, horizon, *, scene_config, project_package=None,
                     headless=None, enable_cameras=None, viewer=None, **registries):
    from shared.assembly.project import prepare_scene
    from mjlab.envs import ManagerBasedRlEnv
    if viewer is not None:
        # Preview asks Isaac to enable camera rendering for its viewport video. MuJoCo draws
        # the window/video itself, so scene cameras keep the scene's own setting.
        enable_cameras = None
    prepared = prepare_scene(parameters, scene_config=scene_config, project_package=project_package,
                             headless=True, enable_cameras=enable_cameras, **registries)
    # headless=False (view) opens MuJoCo's window; otherwise frames are rendered offscreen (record).
    video = Video(viewer) if viewer is not None and headless is not False else None
    window = Window(viewer or {}) if headless is False else None
    class NativeEnv(TerminalObservationMixin, ManagerBasedRlEnv):
        backend_name = 'mjlab'
        def render(self):
            if video is None:
                raise ValueError('Create the mjlab environment with a viewer to render video frames')
            return video.render(self)
        @property
        def window_closed(self):
            return window is not None and window.closed
        def reset(self, *args, **kwargs):
            result = super().reset(*args, **kwargs)
            if window is not None:
                window.show(self)
            return result
        def step(self, actions):
            result = super().step(actions)
            if window is not None:
                window.show(self)
            return result
        def close(self):
            if video is not None:
                video.close()
            if window is not None:
                window.close()
            super().close()
    cfg = build_config(prepared, num_envs=num_envs, seed=seed, horizon=horizon)
    native = NativeEnv(cfg, device=parameters.get('device', 'cuda:0'))
    if native.max_episode_length != horizon:
        native.close()
        raise ValueError('Native episode horizon differs from the RLinf horizon')
    return Environment(native), Owner()
