"""Internal coordinator for the standalone composition preview.

Order: action -> physics -> device systems -> task evaluation -> observation.
Reset order: world -> systems -> action adapter -> task -> observation.
Systems run in declared order, once per step; there is no implicit repeated
event propagation. Policies stay outside this module. A task-free preview
returns zero rewards and no terminations, rather than implying task success.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import Any, Protocol

from shared.assembly.signals import Observation


class System(Protocol):
    """Project-owned device logic, with selected-environment reset support."""
    def reset(self, world, env_ids=None): ...
    def step(self, world, dt: float): ...


class ActionAdapter(Protocol):
    """May map joint commands, call controllers, or advance stateful skills."""
    def reset(self, world, env_ids=None): ...
    def apply(self, world, action): ...


@dataclass
class Evaluation:
    reward: Any
    terminated: Any
    truncated: Any
    metrics: dict


class Task(Protocol):
    def reset(self, world, env_ids=None): ...
    def evaluate(self, world) -> Evaluation: ...


class AssemblyEnv:
    def __init__(self, world, observation: Observation, *, action=None, task=None, systems=()):
        import torch

        self.world, self.observation = world, observation
        self.action, self.task, self.systems = action, task, tuple(systems)
        self.num_envs = world.num_envs
        self.device = torch.device(getattr(world, "device", "cpu"))
        self._ready = False
        self._closed = False

    def _snapshot(self, value):
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: self._snapshot(item) for key, item in value.items()}
        return deepcopy(value)

    def _observe(self):
        import torch

        result = {}
        for name, value in self.observation.read(self.world.signals).items():
            value = torch.as_tensor(value, device=self.device).detach().clone()
            if value.ndim < 1 or value.shape[0] != self.num_envs:
                raise ValueError(f"Observation {name} must have leading dimension num_envs")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"Observation {name} must be finite")
            result[name] = value
        return result

    def reset(self, env_ids=None, *, seed=None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._ready and env_ids is not None:
            raise RuntimeError("First reset must initialize all environments")
        self.world.environment_ids(env_ids)
        if seed is not None:
            self.world.seed(seed)
        self.world.reset(env_ids)
        for component in (*self.systems, self.action, self.task):
            if component is not None:
                component.reset(self.world, env_ids)
        self._ready = True
        return self._observe(), {}

    def step(self, action=None):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if not self._ready:
            raise RuntimeError("Call reset before step")
        if self.action is not None:
            self.action.apply(self.world, action)
        elif action is not None:
            raise ValueError("No action adapter was configured")
        self.world.advance()
        for system in self.systems:
            system.step(self.world, self.world.step_dt)
        evaluation = self.task.evaluate(self.world) if self.task else None
        import torch

        if evaluation is None:
            rewards = torch.zeros(self.num_envs, device=self.device)
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            truncated = torch.zeros_like(terminated)
            info = {}
        else:
            rewards = torch.as_tensor(evaluation.reward, dtype=torch.float32, device=self.device).detach().clone()
            terminated = torch.as_tensor(evaluation.terminated, dtype=torch.bool, device=self.device).clone()
            truncated = torch.as_tensor(evaluation.truncated, dtype=torch.bool, device=self.device).clone()
            if any(value.shape != (self.num_envs,) for value in (rewards, terminated, truncated)):
                raise ValueError("Task outputs must have shape [num_envs]")
            if not torch.isfinite(rewards).all():
                raise ValueError("Task rewards must be finite")
            info = {"metrics": self._snapshot(evaluation.metrics)}
            for name in ("success", "failure"):
                if name in evaluation.metrics:
                    value = torch.as_tensor(evaluation.metrics[name], dtype=torch.bool, device=self.device)
                    if value.shape != (self.num_envs,):
                        raise ValueError(f"Task metric {name} must have shape [num_envs]")
        info["task_enabled"] = self.task is not None
        return self._observe(), rewards, terminated, truncated, info

    def close(self):
        if not self._closed:
            self.world.close()
            self._closed = True
