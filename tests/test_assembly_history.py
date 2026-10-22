"""Project observation selection never advances the native history."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from shared.backends.isaac.environment import Environment


def test_environment_returns_native_observation_without_computing_history_again():
    values = torch.tensor([[1., 2., 3.]])
    native = SimpleNamespace(
        num_envs=1, device="cpu", action_manager=SimpleNamespace(total_action_dim=1),
        observation_manager=SimpleNamespace(compute=Mock()),
        reset=Mock(return_value=({"states": values}, {})),
        step=Mock(return_value=({"states": values}, torch.ones(1),
                               torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {})),
    )
    environment = Environment(native)
    reset, _ = environment.reset()
    stepped, _, _, _, _ = environment.step(torch.zeros(1, 1))
    values.fill_(99)
    for observations in (reset, stepped):
        torch.testing.assert_close(observations["states"], torch.tensor([[1., 2., 3.]]))
    native.observation_manager.compute.assert_not_called()
