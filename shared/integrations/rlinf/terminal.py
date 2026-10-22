"""Terminal snapshots around Isaac Lab's native step/reset, without replacing it.

Isaac Lab 2.3 does not expose pre-reset observations. Only this small hook uses
its observation-manager internals. Observation functions must be read-only;
stateful terms/modifiers/noise models require an explicit snapshot implementation
and are rejected rather than evaluated twice with hidden state changes.
"""

from copy import deepcopy
import random

import numpy as np
import torch


def snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(item) for item in value)
    return deepcopy(value)


def validate_terminal_observations(manager):
    required = ("_group_obs_term_history_buffer", "_obs_buffer",
                "_group_obs_class_instances", "_group_obs_class_term_cfgs")
    if any(not hasattr(manager, name) for name in required):
        raise ValueError("Unsupported Isaac observation manager: terminal snapshot API changed")
    if manager._group_obs_class_instances or any(manager._group_obs_class_term_cfgs.values()):
        raise ValueError("Native terminal snapshots require read-only observation functions; "
                         "stateful observation terms, modifiers and noise models are not yet supported")


def snapshot_terminal_observations(manager, device):
    """Include the newest terminal sample without advancing live history or RNG."""
    histories, cached = manager._group_obs_term_history_buffer, manager._obs_buffer
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        manager._group_obs_term_history_buffer = deepcopy(histories)
        with torch.random.fork_rng(devices=devices):
            return snapshot(manager.compute(update_history=True))
    finally:
        manager._group_obs_term_history_buffer, manager._obs_buffer = histories, cached
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


class TerminalObservationMixin:
    """Native auto-reset is the sole owner of done rows; explicit reset still works."""

    def step(self, actions):
        self._terminal_observation = None
        self._terminal_state = None
        self._capturing_terminal = True
        try:
            observations, rewards, terminated, truncated, extras = super().step(actions)
            info = snapshot(extras)
            info["autoreset"] = True
            if self._terminal_observation is not None:
                info["terminal_observation"] = self._terminal_observation
            if self._terminal_state is not None:
                info["terminal_state"] = self._terminal_state
            return observations, rewards, terminated, truncated, info
        finally:
            self._capturing_terminal = False

    def _reset_idx(self, env_ids):
        if getattr(self, "_capturing_terminal", False) and len(env_ids):
            chain = None
            if 'control_chain' in getattr(self.action_manager, 'active_terms', ()):
                chain = self.action_manager.get_term('control_chain').chain
            state = chain.snapshot_state() if chain is not None else None
            try:
                self._terminal_observation = snapshot_terminal_observations(self.observation_manager, self.device)
            finally:
                if chain is not None:
                    chain.restore_state(state)
            observer = getattr(self, "terminal_state_fn", None)
            if observer is not None:
                self._terminal_state = snapshot(observer())
        return super()._reset_idx(env_ids)
