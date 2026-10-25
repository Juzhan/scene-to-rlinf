"""Isaac environment adapter using RLinf's existing subprocess transport."""

from functools import partial
from copy import deepcopy
import queue
import traceback

import torch
from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv

from .tensors import _make_environment, _prepare_actions, _validate_config, _wrap_observations
from .observations import pack, copy_rows, same_schema

_ENVIRONMENT_FACTORIES = {}


class EnvironmentFailure(RuntimeError):
    """Exception raised inside the Isaac subprocess, sent back as a result."""


class _ReportingEnvironment:
    """Child side: RLinf's worker loop does not catch step/reset errors, so the
    parent would wait forever for a result. Send the traceback instead."""

    def __init__(self, environment):
        self._environment = environment

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def _call(self, method, *args, **kwargs):
        try:
            return method(*args, **kwargs)
        except Exception:
            return EnvironmentFailure(traceback.format_exc()), None

    def reset(self, *args, **kwargs):
        return self._call(self._environment.reset, *args, **kwargs)

    def step(self, *args, **kwargs):
        return self._call(self._environment.step, *args, **kwargs)


def _make_reporting_environment(*args, **kwargs):
    environment, owner = _make_environment(*args, **kwargs)
    return _ReportingEnvironment(environment), owner


class _LiveQueue:
    """Parent side: stop waiting when the subprocess died without a result."""

    def __init__(self, results, process):
        self._results, self._process = results, process

    def get(self):
        while True:
            try:
                return self._results.get(timeout=5)
            except queue.Empty:
                if not self._process.is_alive():
                    raise RuntimeError(
                        f"Isaac environment process exited with code {self._process.exitcode}; "
                        "see its log above") from None

    def __getattr__(self, name):
        return getattr(self._results, name)


class _CheckedTransport:
    def __init__(self, transport):
        self._transport = transport
        process = getattr(transport, "isaac_lab_process", None)
        if process is not None:
            transport.obs_queue = _LiveQueue(transport.obs_queue, process)

    def __getattr__(self, name):
        return getattr(self._transport, name)

    @staticmethod
    def _checked(result):
        if isinstance(result, tuple) and result and isinstance(result[0], EnvironmentFailure):
            raise EnvironmentFailure("Isaac environment failed in its subprocess:\n" + str(result[0]))
        return result

    def reset(self, *args, **kwargs):
        return self._checked(self._transport.reset(*args, **kwargs))

    def step(self, *args, **kwargs):
        return self._checked(self._transport.step(*args, **kwargs))


class IsaacAssemblyRLinfEnv(IsaaclabBaseEnv):
    """RLinf tracks rollouts while Isaac Lab owns episode reset.

    Native environments mark steps with ``autoreset=True`` and provide the
    full-batch terminal observation before reset, for value bootstrapping.
    """

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.factory = _ENVIRONMENT_FACTORIES.get(cfg.init_params.id)
        _validate_config(cfg, factory=self.factory)
        self._closed = False
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

    def _make_env_function(self):
        return partial(_make_reporting_environment, self.cfg.init_params, self.num_envs,
                       self.seed, self.cfg.max_episode_steps, factory=self.factory)

    def _init_isaaclab_env(self):
        from rlinf.envs.sim.isaaclab import isaaclab_env

        self.env = _CheckedTransport(isaaclab_env.SubProcIsaacLabEnv(self._make_env_function()))
        self.env.reset(seed=self.seed)

    def _wrap_obs(self, observations):
        return pack(_wrap_observations(observations, self.num_envs), self.num_envs,
                    self.task_description)

    def step(self, actions=None, auto_reset=True):
        if actions is not None:
            actions = _prepare_actions(actions, (self.num_envs, self.cfg.init_params.action_dim), self.device)
        obs, step_reward, terminations, truncations, infos = self.env.step(actions)
        step_reward, terminations, truncations = (
            value.clone() for value in (step_reward, terminations, truncations)
        )
        obs = self._wrap_obs(obs)
        infos = deepcopy(infos)
        native_autoreset = infos.get("autoreset") is True
        native_dones = terminations | truncations
        self._elapsed_steps += 1
        truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) | truncations
        dones = terminations | truncations
        if native_autoreset and (dones & ~native_dones).any():
            raise RuntimeError(
                "RLinf max_episode_steps ended an episode before native autoreset; "
                "configure the native timeout to match the RLinf horizon"
            )

        terminal_obs = None
        if native_autoreset and dones.any():
            if "terminal_observation" not in infos:
                raise RuntimeError("Native autoreset requires terminal_observation before reset")
            terminal_obs = self._wrap_obs(infos.pop("terminal_observation"))
            if not same_schema(terminal_obs, obs):
                raise ValueError("terminal_observation must match the full-batch observation schema")

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            # A termination may be a configured failure, not task success.
            infos["episode"]["success_at_end"] = torch.as_tensor(
                infos.get("metrics", {}).get("success", terminations),
                device=self.device, dtype=torch.bool).clone()
            terminations[:] = False

        if terminal_obs is not None:
            # A native reset already happened even when chunk_step passes
            # auto_reset=False. Preserve terminal data and reset only metrics.
            final_info = deepcopy(infos)
            self._reset_metrics(torch.nonzero(dones, as_tuple=False).flatten())
            infos = dict(
                autoreset=True, final_observation=terminal_obs, final_info=final_info,
                _final_info=dones.clone(), _final_observation=dones.clone(),
                _elapsed_steps=dones.clone(),
            )
        elif dones.any() and auto_reset and self.auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        chunk = getattr(self, "_chunk_state", None)
        if chunk is not None:
            active = ~chunk["done"]
            step_reward = torch.where(active, step_reward, 0.0)
            terminations = terminations & active
            truncations = truncations & active
            ended = terminations | truncations
            chunk["native_end"] = native_dones & native_autoreset
            if ended.any():
                final_obs = infos.get("final_observation", obs)
                episode = infos.get("final_info", infos)["episode"]
                if chunk["observation"] is None:
                    chunk["observation"] = deepcopy(final_obs)
                    chunk["episode"] = deepcopy(episode)
                copy_rows(chunk["observation"], final_obs, ended)
                for name, value in episode.items():
                    chunk["episode"][name][ended] = value[ended]
                for name, value in infos.get("final_info", infos).get("metrics", {}).items():
                    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.num_envs:
                        chunk["metrics"].setdefault(name, value.clone())[ended] = value[ended]
            chunk["done"] |= ended
            for name in ("_final_info", "_final_observation", "_elapsed_steps"):
                if name in infos:
                    infos[name] = infos[name] & active
        return obs, step_reward, terminations, truncations, infos

    def _handle_auto_reset(self, dones, final_obs, infos):
        chunk = getattr(self, "_chunk_state", None)
        if chunk is not None:
            terminal_obs = deepcopy(final_obs)
            final_info = deepcopy(infos.get("final_info", infos))
            copy_rows(terminal_obs, chunk["observation"], dones)
            for name, value in chunk["episode"].items():
                final_info["episode"][name][dones] = value[dones]
            for name, value in chunk["metrics"].items():
                final_info.setdefault("metrics", {}).setdefault(name, value.clone())[dones] = value[dones]
            reset_ids = torch.where(dones & ~chunk["native_end"])[0]
            if reset_ids.numel():
                final_obs, _ = self.reset(env_ids=reset_ids)
            return final_obs, dict(
                autoreset=True, final_observation=terminal_obs, final_info=final_info,
                _final_info=dones.clone(), _final_observation=dones.clone(),
                _elapsed_steps=dones.clone(),
            )
        if infos.get("autoreset") is True:
            return final_obs, infos
        return super()._handle_auto_reset(dones, final_obs, infos)

    def _record_metrics(self, step_reward, terminations, infos):
        success = self.success_once.clone()
        failure = self.fail_once.clone()
        metrics = infos.get("metrics", {})
        if "success" in metrics:
            success |= torch.as_tensor(metrics["success"], device=self.device, dtype=torch.bool)
        if "failure" in metrics:
            failure |= torch.as_tensor(metrics["failure"], device=self.device, dtype=torch.bool)
        infos = super()._record_metrics(step_reward, terminations, infos)
        self.success_once.copy_(success)
        self.fail_once.copy_(failure)
        infos["episode"].update(success_once=success, fail_once=failure)
        # Task metrics at each episode's last step, logged by RLinf as env/task/<name>.
        for name, entry in infos.get("task_metrics", {}).items():
            value = torch.as_tensor(entry["value"], device=self.device)
            if value.shape == (self.num_envs,):
                valid = torch.as_tensor(entry["valid"], device=self.device, dtype=torch.bool)
                infos["episode"]["task/" + name] = torch.where(valid, value.float(), 0.)
        return infos

    def chunk_step(self, actions):
        try:
            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError("Actions must have shape [num_envs, chunk_length, action_dim]") from error
        if actions.ndim != 3 or actions.shape[1] < 1:
            raise ValueError("Actions must have shape [num_envs, positive chunk_length, action_dim]")
        actions = _prepare_actions(
            actions, (self.num_envs, actions.shape[1], self.cfg.init_params.action_dim), self.device)
        self._chunk_state = {
            "done": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "native_end": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "observation": None, "episode": None, "metrics": {},
        }
        try:
            return super().chunk_step(actions)
        finally:
            self._chunk_state = None

    def close(self):
        if not self._closed:
            self._closed = True
            super().close()


def get_isaac_assembly_factory(task_id):
    return _ENVIRONMENT_FACTORIES.get(task_id)


def register_isaac_assembly_env(task_id, *, factory):
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS

    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a nonempty string")
    registered = REGISTER_ISAACLAB_ENVS.get(task_id)
    if registered is not None and registered is not IsaacAssemblyRLinfEnv:
        raise ValueError(f"Task {task_id} is already registered")
    if not callable(factory):
        raise TypeError("Environment factory must be callable")
    previous = _ENVIRONMENT_FACTORIES.get(task_id)
    if previous is not None and previous != factory:
        raise ValueError(f"Task {task_id} already has an environment factory")
    _ENVIRONMENT_FACTORIES[task_id] = factory
    REGISTER_ISAACLAB_ENVS[task_id] = IsaacAssemblyRLinfEnv
