"""State observations and terminal states over RLinf's native Isaac subprocess."""

from functools import partial
from importlib import import_module
import os

import torch
from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv


def resolve(entry):
    module, symbol = entry.split(":", 1)
    return getattr(import_module(module), symbol)


class ProcessExit:
    def __init__(self, application):
        self.application = application

    def close(self):
        import sys
        import traceback

        # Shutdown is also called while unwinding construction failures. Keep
        # the original traceback and a failing status instead of hiding them.
        exception = sys.exc_info()
        if exception[0] is not None:
            traceback.print_exception(*exception)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if exception[0] is not None else 0)


def make_simulation(parameters, num_envs, seed, horizon, *, headless=True, enable_cameras=False, viewer=None):
    if headless:
        os.environ.pop("DISPLAY", None)
    elif not os.environ.get("DISPLAY"):
        raise RuntimeError("GUI preview requires DISPLAY; run from a desktop terminal or use record")
    from isaaclab.app import AppLauncher

    application = AppLauncher(headless=headless, enable_cameras=enable_cameras, fast_shutdown=False).app
    from isaaclab.envs import ManagerBasedRLEnv

    class TerminalStateEnv(ManagerBasedRLEnv):
        def step(self, actions):
            self.final_states = None
            self.stepping = True
            try:
                observation, reward, terminated, truncated, info = super().step(actions)
                info = dict(info)
                if self.final_states is not None:
                    info["terminal_states"] = self.final_states
                return observation, reward, terminated, truncated, info
            finally:
                self.stepping = False

        def _reset_idx(self, env_ids):
            if getattr(self, "stepping", False):
                self.final_states = self.observation_manager.compute(update_history=False)["policy"].clone()
            super()._reset_idx(env_ids)

    config = resolve(parameters.env_cfg)()
    config.scene.num_envs = num_envs
    config.seed = seed
    config.sim.device = parameters.get("device", "cuda:0")
    config.episode_length_s = horizon * config.sim.dt * config.decimation
    config.observations.policy.concatenate_terms = True
    if parameters.get("configure"):
        resolve(parameters.configure)(config, parameters)
    if viewer is not None:
        config.scene.clone_in_fabric = False
        config.viewer.origin_type = "env"
        config.viewer.env_index = 0
        for name, value in viewer.items():
            setattr(config.viewer, name, value)
    environment = TerminalStateEnv(cfg=config, render_mode="rgb_array" if enable_cameras else None)
    return environment, ProcessExit(application)


class IsaacStateEnv(IsaaclabBaseEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        if not cfg.auto_reset or cfg.ignore_terminations or cfg.use_rel_reward:
            raise ValueError("State tasks preserve native rewards/terminations and require auto_reset=True")
        if cfg.video_cfg.save_video:
            raise ValueError("This state adapter has no camera; use a visual task adapter for recording")
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _make_env_function(self):
        return partial(make_simulation, self.cfg.init_params, self.num_envs, self.seed, self.cfg.max_episode_steps)

    def _wrap_obs(self, observations):
        states = observations["policy"]
        expected = (self.num_envs, self.cfg.init_params.obs_dim)
        if tuple(states.shape) != expected or not torch.isfinite(states).all():
            raise ValueError(f"Expected finite policy states of shape {expected}")
        return {"states": states.clone()}

    def step(self, actions=None, auto_reset=True):
        observations, rewards, terminated, truncated, raw_info = self.env.step(actions)
        observations = self._wrap_obs(observations)
        rewards, terminated, truncated = rewards.clone(), terminated.bool().clone(), truncated.bool().clone()
        self._elapsed_steps += 1
        self.returns += rewards
        episode = {"return": self.returns.clone(), "episode_len": self._elapsed_steps.clone(),
                   "reward": self.returns / self._elapsed_steps}
        info = {"episode": episode}
        dones = terminated | truncated
        if dones.any():
            info.update(final_observation={"states": raw_info["terminal_states"].clone()},
                        final_info={"episode": episode}, _final_observation=dones.clone(),
                        _final_info=dones.clone(), _elapsed_steps=dones.clone())
            self._reset_metrics(torch.where(dones)[0])
        return observations, rewards, terminated, truncated, info

    def _handle_auto_reset(self, dones, observations, infos):
        return observations, infos

    def chunk_step(self, actions):
        if tuple(actions.shape) != (self.num_envs, 1, self.cfg.init_params.action_dim):
            raise ValueError("Isaac state tasks require [num_envs, 1, action_dim]; VLA uses its own adapter")
        return super().chunk_step(actions)
