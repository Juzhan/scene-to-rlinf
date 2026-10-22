"""The trainer adapter accepts an environment, without requiring runtime classes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from shared.integrations.rlinf.tensors import _make_environment


def environment():
    return SimpleNamespace(
        num_envs=3, device="cpu", action_dim=2,
        reset=Mock(), step=Mock(), close=Mock(),
    )


def test_environment_protocol_does_not_require_world_task_or_action_classes():
    env = environment()
    parameters = SimpleNamespace(action_dim=2)
    result, owner = _make_environment(parameters, 3, 4, 20, factory=lambda *args: (env, None))
    assert result is env and owner is None
    assert not hasattr(result, "world") and not hasattr(result, "task")
    env.close.assert_not_called()


@pytest.mark.parametrize("field,value,message", [
    ("step", None, "step"),
    ("device", "unknown-device", "device"),
    ("num_envs", 2, "num_envs"),
    ("action_dim", 1, "action_dim"),
    ("action_dim", True, "positive integer"),
])
def test_environment_protocol_rejects_mismatches_and_releases_resources(field, value, message):
    env = environment()
    setattr(env, field, value)
    owner = SimpleNamespace(close=Mock())
    with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
        _make_environment(SimpleNamespace(action_dim=2), 3, 4, 20, factory=lambda *args: (env, owner))
    env.close.assert_called_once()
    owner.close.assert_called_once()


def test_registration_can_repeat_the_same_project_bound_method(monkeypatch):
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
    from shared.integrations.rlinf.isaac import _ENVIRONMENT_FACTORIES, register_isaac_assembly_env

    class Project:
        def make_environment(self, *args):
            return environment(), None

    project = Project()
    identifier = "test-bound-project-registration"
    monkeypatch.setitem(_ENVIRONMENT_FACTORIES, identifier, None)
    monkeypatch.setitem(REGISTER_ISAACLAB_ENVS, identifier, None)
    register_isaac_assembly_env(identifier, factory=project.make_environment)
    register_isaac_assembly_env(identifier, factory=project.make_environment)
    with pytest.raises(ValueError, match="already has"):
        register_isaac_assembly_env(identifier, factory=Project().make_environment)
