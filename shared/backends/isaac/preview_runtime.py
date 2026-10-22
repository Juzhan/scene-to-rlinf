"""Shared runtime state and lifecycle contract, without a physics engine import."""

from abc import ABC, abstractmethod
from copy import deepcopy
from math import isfinite
from operator import index
from random import Random

from shared.assembly.signals import Signal, Signals


class RuntimeWorld(ABC):
    """Backends implement physics; descriptions and semantic state stay shared."""

    def __init__(self, description, *, num_envs, step_dt, prepare_signals=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isfinite(step_dt) or step_dt <= 0:
            raise ValueError("step_dt must be finite and positive")
        self.description = deepcopy(description)
        self.index = self.description.index()
        self.num_envs, self.step_dt = num_envs, step_dt
        self.rng = Random()
        self.signals = Signals(prepare=prepare_signals)
        self.semantic_state = {path: [deepcopy(node.initial_state) for _ in range(num_envs)]
                               for path, (node, _) in self.index.items()}
        for path, (node, _) in self.index.items():
            self.signals.register(path + "/parameters", Signal(lambda entity=node: deepcopy(entity.parameters), "parameter"))
            self.signals.register(path + "/tags", Signal(lambda entity=node: entity.tags, "metadata"))
            self.signals.register(path + "/semantic_state", Signal(lambda source=path: deepcopy(self.semantic_state[source]), "semantic"))
            for key in node.parameters:
                self.signals.register(path + "/parameters/" + key, Signal(lambda entity=node, field=key: deepcopy(entity.parameters[field]), "parameter"))
            for key in node.initial_state:
                self.signals.register(path + "/state/" + key, Signal(lambda source=path, field=key: deepcopy(
                    [state[field] for state in self.semantic_state[source]]), "semantic"))

    def seed(self, seed):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.rng.seed(seed)

    def close(self):
        """Application ownership and shutdown remain with the backend entry point."""

    def environment_ids(self, env_ids=None):
        if env_ids is None:
            return list(range(self.num_envs))
        try:
            values = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
            if any(isinstance(value, bool) for value in values):
                raise TypeError("Boolean environment indices are not supported")
            ids = [index(value) for value in values]
        except (TypeError, ValueError):
            raise ValueError("env_ids must be a one-dimensional integer sequence") from None
        if len(set(ids)) != len(ids) or any(value < 0 or value >= self.num_envs for value in ids):
            raise ValueError("env_ids must contain distinct valid indices")
        return ids

    def set_state(self, path, key, value, env_ids=None):
        """Write declared semantic state only, never a physical object pose."""
        if key not in self.index[path][0].initial_state:
            raise KeyError(f"Undeclared semantic state: {path}/{key}")
        for env_id in self.environment_ids(env_ids):
            self.semantic_state[path][env_id][key] = deepcopy(value)

    def reset_semantic_state(self, env_ids=None):
        ids = self.environment_ids(env_ids)
        for path, (node, _) in self.index.items():
            for env_id in ids:
                self.semantic_state[path][env_id] = deepcopy(node.initial_state)

    def register_signal(self, path, signal):
        self.signals.register(path, signal)

    def set_joint_position_target(self, path, target, env_ids=None):
        """Target shape: [selected environments, joints], in joint_names order.

        Units are radians/metres as declared by joint_position. A backend must
        reject unsupported control instead of silently changing its meaning.
        """
        raise NotImplementedError("This backend does not support joint-position targets")

    @abstractmethod
    def advance(self):
        """Advance one control step; update physics signals before returning."""

    @abstractmethod
    def reset(self, env_ids=None):
        """Restore physics, control and semantic state for selected environments."""