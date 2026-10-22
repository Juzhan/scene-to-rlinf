"""Run the installed Isaac Lab lifecycle on CPU fakes, without importing Kit.

The SDK methods are compiled from their installed source, rather than copied
here. This catches changes in their ordering and in history-buffer semantics.
"""

import ast
from collections.abc import Sequence
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from shared.integrations.rlinf.terminal import (
    TerminalObservationMixin,
    snapshot_terminal_observations,
    validate_terminal_observations,
)


def _installed_source():
    spec = find_spec("isaaclab")
    if spec is None or spec.origin is None:
        pytest.skip("Isaac Lab source is unavailable; no simulator is required")
    package = Path(spec.origin).parent
    candidates = (package / "source" / "isaaclab" / "isaaclab", package)
    for candidate in candidates:
        if (candidate / "envs" / "manager_based_rl_env.py").is_file():
            return candidate
    pytest.skip("Installed Isaac Lab package does not expose its Python source")


def _sdk_class(path, name, *, methods=None, namespace=None):
    tree = ast.parse(path.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    if methods is not None:
        original.body = [node for node in original.body
                         if isinstance(node, ast.FunctionDef) and node.name in methods]
    original.bases = []
    original.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), original], type_ignores=[])
    context = {"torch": torch, "Sequence": Sequence, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), context)
    return context[name]


@pytest.fixture(scope="module")
def sdk():
    source = _installed_source()
    buffer = _sdk_class(source / "utils" / "buffers" / "circular_buffer.py", "CircularBuffer")
    noise = SimpleNamespace(NoiseCfg=type("NoiseCfg", (), {}), NoiseModelCfg=type("NoiseModelCfg", (), {}))
    observation = _sdk_class(source / "managers" / "observation_manager.py", "ObservationManager",
        methods={"compute", "compute_group"}, namespace={"CircularBuffer": buffer, "noise": noise})
    step = _sdk_class(source / "envs" / "manager_based_rl_env.py", "ManagerBasedRLEnv", methods={"step"})
    reset = _sdk_class(source / "envs" / "manager_based_env.py", "ManagerBasedEnv", methods={"reset"})
    return SimpleNamespace(buffer=buffer, observation=observation, step=step, reset=reset, noise=noise)


def _manager(sdk, world, *, noisy=False):
    manager = sdk.observation()
    manager._env = world
    manager._group_obs_term_names = {"policy": ["state"]}
    noise = None
    if noisy:
        noise = sdk.noise.NoiseCfg()
        noise.func = lambda value, cfg: value + torch.rand_like(value)
    manager._group_obs_term_cfgs = {"policy": [SimpleNamespace(
        func=lambda env: env.values[:, None], params={}, modifiers=None, noise=noise,
        clip=None, scale=None, history_length=3, flatten_history_dim=True)]}
    manager._group_obs_concatenate = {"policy": True}
    manager._group_obs_concatenate_dim = {"policy": 1}
    history = sdk.buffer(max_len=3, batch_size=world.num_envs, device="cpu")
    for sample in ((1., 10.), (2., 20.), (3., 30.)):
        history.append(torch.tensor(sample)[:, None])
    manager._group_obs_term_history_buffer = {"policy": {"state": history}}
    manager._group_obs_class_term_cfgs = {"policy": []}
    manager._group_obs_class_instances = []
    manager._obs_buffer = {"policy": history.buffer.reshape(world.num_envs, -1)}
    return manager


def _environment(sdk, *, terminal=True, timeout=False, recorder=True):
    class Native(sdk.step, sdk.reset):
        def _reset_idx(self, env_ids):
            self.events.append(("reset", env_ids.tolist()))
            self.values[env_ids] = -5.
            self.episode_length_buf[env_ids] = 0
            self.observation_manager._group_obs_term_history_buffer["policy"]["state"].reset(env_ids)

    class Environment(TerminalObservationMixin, Native):
        pass

    env = Environment()
    env.num_envs, env.device = 2, "cpu"
    env.step_dt, env.physics_dt = .02, .01
    env.cfg = SimpleNamespace(decimation=2, sim=SimpleNamespace(render_interval=2),
                              num_rerenders_on_reset=1, wait_for_textures=False)
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.values = torch.tensor([3., 30.])
    env.events, env.extras = [], {}
    env.observation_manager = _manager(sdk, env)
    env.action_manager = SimpleNamespace(
        total_action_dim=1,
        process_action=lambda action: env.events.append("process_action"),
        apply_action=lambda: env.events.append("apply_action"))
    env.scene = SimpleNamespace(write_data_to_sim=lambda: env.events.append("write"),
                                update=lambda dt: env.events.append("scene_update"))

    def physics_step(render):
        env.events.append("physics")
        env.values += 1

    env.sim = SimpleNamespace(has_gui=lambda: False, has_rtx_sensors=lambda: True,
        step=physics_step, render=lambda: env.events.append("render"),
        forward=lambda: env.events.append("forward"))
    env.recorder_manager = SimpleNamespace(active_terms=["sample"] if recorder else [],
        record_pre_step=lambda: env.events.append("record_pre_step"),
        record_post_physics_decimation_step=lambda: env.events.append("record_physics"),
        record_post_step=lambda: env.events.append("record_post_step"),
        record_pre_reset=lambda ids: env.events.append(("record_pre_reset", ids.tolist())),
        record_post_reset=lambda ids: env.events.append(("record_post_reset", ids.tolist())))
    env.termination_manager = SimpleNamespace(
        terminated=torch.tensor([terminal and not timeout, False]),
        time_outs=torch.tensor([terminal and timeout, False]))
    env.termination_manager.compute = lambda: (
        env.termination_manager.terminated | env.termination_manager.time_outs)
    env.reward_manager = SimpleNamespace(compute=lambda dt: torch.tensor([1., 2.]))
    env.command_manager = SimpleNamespace(compute=lambda dt: env.events.append("command"))
    env.event_manager = SimpleNamespace(available_modes=["interval"],
        apply=lambda **kwargs: env.events.append("interval"))
    return env


@pytest.mark.parametrize("timeout", [False, True])
def test_native_step_preserves_recorder_order_and_terminal_history(sdk, timeout):
    env = _environment(sdk, timeout=timeout)
    obs, reward, terminated, truncated, info = env.step(torch.zeros(2, 1))
    assert info["autoreset"] is True
    assert terminated.tolist() == [not timeout, False]
    assert truncated.tolist() == [timeout, False]
    assert reward.tolist() == [1., 2.]
    assert env.events == ["process_action", "record_pre_step", "apply_action", "write", "physics",
        "record_physics", "scene_update", "apply_action", "write", "physics", "record_physics",
        "render", "scene_update", "record_post_step", ("record_pre_reset", [0]), ("reset", [0]),
        "render", ("record_post_reset", [0]), "command", "interval"]
    torch.testing.assert_close(info["terminal_observation"]["policy"],
                               torch.tensor([[2., 3., 5.], [20., 30., 32.]]))
    torch.testing.assert_close(obs["policy"], torch.tensor([[-5., -5., -5.], [20., 30., 32.]]))
    assert env.episode_length_buf.tolist() == [0, 1]
    saved = info["terminal_observation"]["policy"].clone()
    env.values[:] = 100.
    env.observation_manager.compute(update_history=True)
    torch.testing.assert_close(info["terminal_observation"]["policy"], saved)


def test_native_nonterminal_step_does_not_capture_or_reset(sdk):
    env = _environment(sdk, terminal=False, recorder=False)
    obs, _, _, _, info = env.step(torch.zeros(2, 1))
    assert info["autoreset"] is True
    assert "terminal_observation" not in info
    assert not any(isinstance(event, tuple) for event in env.events)
    assert "record_post_step" not in env.events
    torch.testing.assert_close(obs["policy"], torch.tensor([[2., 3., 5.], [20., 30., 32.]]))


def test_explicit_native_reset_does_not_capture_terminal_observation(sdk):
    env = _environment(sdk)
    original = env.observation_manager.compute
    calls = []
    def compute(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    env.observation_manager.compute = compute
    obs, info = env.reset(env_ids=torch.tensor([0]))
    assert calls == [{"update_history": True}]
    assert "terminal_observation" not in info
    assert env.events.count(("reset", [0])) == 1
    torch.testing.assert_close(obs["policy"][0], torch.tensor([-5., -5., -5.]))


def test_terminal_snapshot_preserves_buffers_cache_and_rng(sdk):
    world = SimpleNamespace(num_envs=2, values=torch.tensor([5., 32.]))
    manager = _manager(sdk, world, noisy=True)
    buffers, cached = manager._group_obs_term_history_buffer, manager._obs_buffer
    before = buffers["policy"]["state"].buffer.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        rng = torch.random.get_rng_state().clone()
        terminal = snapshot_terminal_observations(manager, "cpu")
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert manager._group_obs_term_history_buffer is buffers
        assert manager._obs_buffer is cached
        torch.testing.assert_close(buffers["policy"]["state"].buffer, before)
        actual = manager.compute(update_history=True)
        torch.testing.assert_close(terminal["policy"], actual["policy"])


def test_terminal_snapshot_restores_state_after_observation_error(sdk):
    world = SimpleNamespace(num_envs=2, values=torch.tensor([5., 32.]))
    manager = _manager(sdk, world)
    buffers, cached = manager._group_obs_term_history_buffer, manager._obs_buffer
    before = buffers["policy"]["state"].buffer.clone()
    def broken_compute(update_history=False):
        manager._group_obs_term_history_buffer["policy"]["state"].append(torch.rand(2, 1))
        manager._obs_buffer = {"partial": torch.ones(2, 1)}
        raise RuntimeError("observation failed")
    manager.compute = broken_compute
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(91)
        rng = torch.random.get_rng_state().clone()
        with pytest.raises(RuntimeError, match="observation failed"):
            snapshot_terminal_observations(manager, "cpu")
        assert torch.equal(torch.random.get_rng_state(), rng)
    assert manager._group_obs_term_history_buffer is buffers
    assert manager._obs_buffer is cached
    torch.testing.assert_close(buffers["policy"]["state"].buffer, before)


@pytest.mark.parametrize("attribute,value", [
    ("_group_obs_class_instances", [object()]),
    ("_group_obs_class_term_cfgs", {"policy": [object()]}),
])
def test_stateful_observation_components_require_explicit_support(sdk, attribute, value):
    manager = _manager(sdk, SimpleNamespace(num_envs=2, values=torch.zeros(2)))
    validate_terminal_observations(manager)
    setattr(manager, attribute, value)
    with pytest.raises(ValueError, match="(?i)stateful"):
        validate_terminal_observations(manager)


def test_native_terminal_callback_reads_physical_state_before_reset(sdk):
    native = _environment(sdk)
    callback = Mock(side_effect=lambda: {"physical": native.values[:, None]})
    native.terminal_state_fn = callback
    _, _, _, _, info = native.step(torch.zeros(2, 1))
    callback.assert_called_once_with()
    torch.testing.assert_close(info["terminal_state"]["physical"], torch.tensor([[5.], [32.]]))
    assert native.values[0].item() == -5.
    native.values[:] = 100.
    torch.testing.assert_close(info["terminal_state"]["physical"], torch.tensor([[5.], [32.]]))


def test_environment_wrapper_leaves_actions_rewards_and_resets_to_native_sdk(sdk):
    from shared.backends.isaac.environment import Environment

    native = _environment(sdk)
    native.close = Mock()
    env = Environment(native)
    assert env.native is native and env.action_dim == 1
    assert not hasattr(env, "world") and not hasattr(env, "task")
    env.reset()
    native.events.clear()
    native.values = torch.tensor([3., 30.])
    native.observation_manager = _manager(sdk, native)
    native.action_manager.process_action = Mock()
    native.reward_manager.compute = Mock(return_value=torch.tensor([1., 2.]))

    action = torch.full((2, 1), 0.25)
    obs, rewards, terminated, truncated, info = env.step(action)

    native.action_manager.process_action.assert_called_once_with(action)
    native.reward_manager.compute.assert_called_once_with(dt=native.step_dt)
    assert native.events.count(("reset", [0])) == 1
    assert set(obs) == set(info["terminal_observation"]) == {"policy"}
    assert info["autoreset"] is True
    torch.testing.assert_close(obs["policy"], torch.tensor([[-5., -5., -5.], [20., 30., 32.]]))
    torch.testing.assert_close(info["terminal_observation"]["policy"],
                               torch.tensor([[2., 3., 5.], [20., 30., 32.]]))
    assert rewards.tolist() == [1., 2.]
    assert terminated.tolist() == [True, False]
    assert truncated.tolist() == [False, False]

    native.termination_manager.terminated[:] = False
    _, _, _, _, next_info = env.step(action)
    assert "terminal_observation" not in next_info
    # SDK buffers can change in a later step without changing emitted data.
    assert terminated.tolist() == [True, False]
    torch.testing.assert_close(obs["policy"], torch.tensor([[-5., -5., -5.], [20., 30., 32.]]))
    env.close()
    native.close.assert_called_once()
