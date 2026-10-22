"""RLinf episode contracts against a CPU transport, without importing Isaac."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf


class CpuTransport:
    """Mimic shared simulator buffers and native per-row reset in step()."""

    def __init__(self, frames, *, native=True, terminal=True):
        self.frames = iter(frames)
        self.native, self.terminal = native, terminal
        self.states = torch.zeros(3, 1)
        self.terminal_buffer = torch.zeros_like(self.states)
        self.reset_calls = []
        self.native_reset_calls = []
        self.actions = []

    def device(self):
        return torch.device("cpu")

    def reset(self, *, seed=None, env_ids=None):
        ids = torch.arange(3) if env_ids is None else env_ids
        self.reset_calls.append(ids.tolist())
        self.states[ids] = 0
        return {"states": self.states}, {}

    def step(self, actions):
        self.actions.append(actions.clone())
        frame = next(self.frames)
        self.states.add_(1)
        terminated = torch.tensor(frame.get("terminated", [False] * 3))
        truncated = torch.tensor(frame.get("truncated", [False] * 3))
        infos = {"metrics": {
            name: torch.tensor(frame.get(name, [False] * 3))
            for name in ("success", "failure")
        }}
        if "task_metrics" in frame:
            infos["task_metrics"] = {name: {"value": torch.tensor(value), "valid": torch.ones(3, dtype=torch.bool)}
                                     for name, value in frame["task_metrics"].items()}
        if self.native:
            infos["autoreset"] = True
            dones = terminated | truncated
            if dones.any():
                self.terminal_buffer.copy_(self.states)
                if self.terminal:
                    infos["terminal_observation"] = {"states": self.terminal_buffer}
                self.native_reset_calls.append(torch.where(dones)[0].tolist())
                self.states[dones] = 0
        return {"states": self.states}, torch.tensor(frame["rewards"]), terminated, truncated, infos

    def close(self):
        pass


@pytest.fixture
def make_environment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "third_party/RLinf"))
    from rlinf.envs.sim.isaaclab import isaaclab_env
    from shared.integrations.rlinf.isaac import IsaacAssemblyRLinfEnv, _ENVIRONMENT_FACTORIES

    def make(transport, *, horizon=10):
        cfg = OmegaConf.create({
            "seed": 3, "auto_reset": True, "ignore_terminations": False,
            "use_rel_reward": False, "max_episode_steps": horizon,
            "video_cfg": {"save_video": False}, "init_params": {
                "id": "cpu-native-reset-contract", "task_description": "CPU test", "action_dim": 1,
            },
        })
        monkeypatch.setitem(_ENVIRONMENT_FACTORIES, cfg.init_params.id, lambda *args: None)
        monkeypatch.setattr(isaaclab_env, "SubProcIsaacLabEnv", lambda factory: transport)
        environment = IsaacAssemblyRLinfEnv(cfg, 3, 0, 1, None)
        # Keep actual BaseEnv counters/metrics, observing only reset invocations.
        environment._reset_metrics = Mock(wraps=environment._reset_metrics)
        return environment

    return make


def advance(environment, mode):
    actions = torch.zeros(3, 1)
    if mode == "chunk":
        observations, rewards, terminated, truncated, infos = environment.chunk_step(actions[:, None])
        assert rewards.shape == terminated.shape == truncated.shape == (3, 1)
        return observations[0], rewards[:, 0], terminated[:, 0], truncated[:, 0], infos[0]
    return environment.step(actions, auto_reset=mode != "step_no_reset")


def test_configured_failure_is_not_reported_as_success_when_terminations_ignored(make_environment):
    transport = CpuTransport([{'rewards': [0., 0., 0.],
        'terminated': [True, True, False], 'success': [True, False, False],
        'failure': [False, True, False]}])
    environment = make_environment(transport)
    environment.ignore_terminations = True
    _, _, _, _, infos = environment.step(torch.zeros(3, 1))
    episode = infos['final_info']['episode']
    assert episode['success_at_end'].tolist() == [True, False, False]
    assert episode['success_once'].tolist() == [True, False, False]
    assert episode['fail_once'].tolist() == [False, True, False]


def test_task_metrics_reach_episode_statistics(make_environment):
    transport = CpuTransport([{'rewards': [0., 0., 0.], 'terminated': [True, False, False],
                               'task_metrics': {'held_s': [2., .5, 0.]}}])
    _, _, _, _, infos = make_environment(transport).step(torch.zeros(3, 1))
    assert infos['final_info']['episode']['task/held_s'].tolist() == [2., .5, 0.]


PARTIAL_END = {
    "rewards": [1., 2., 3.], "terminated": [True, False, False],
    "truncated": [False, True, False], "success": [True, False, False],
    "failure": [False, True, False],
}


@pytest.mark.parametrize("native", [True, False])
def test_native_multistep_chunk_keeps_early_terminal_rows(make_environment, native):
    transport = CpuTransport([
        PARTIAL_END,
        {"rewards": [4., 5., 6.]},
        {"rewards": [7., 8., 9.]},
    ], native=native)
    environment = make_environment(transport)
    observations, rewards, terminated, truncated, infos = environment.chunk_step(torch.zeros(3, 3, 1))
    assert len(observations) == len(infos) == 3
    assert rewards.shape == terminated.shape == truncated.shape == (3, 3)
    assert terminated[:, -1].tolist() == [True, False, False]
    assert truncated[:, -1].tolist() == [False, True, False]
    assert not terminated[:, :-1].any() and not truncated[:, :-1].any()
    assert infos[-1]["_final_observation"].tolist() == [True, True, False]
    assert infos[-1]["final_observation"]["states"].flatten().tolist() == [1., 1., 3.]
    assert infos[-1]["final_info"]["episode"]["return"].tolist() == [1., 2., 18.]
    assert rewards.tolist() == [[1., 0., 0.], [2., 0., 0.], [3., 6., 9.]]
    assert observations[-1]["states"].flatten().tolist() == [0., 0., 3.]
    assert transport.reset_calls == [[0, 1, 2], [0, 1]]
    assert environment._reset_metrics.call_count == (2 if native else 1)
    assert environment.elapsed_steps.tolist() == [0, 0, 3]
    assert environment.returns.tolist() == [0., 0., 18.]


@pytest.mark.parametrize("length", [1, 2, 4])
@pytest.mark.parametrize("native", [True, False])
def test_chunk_executes_actions_in_order_without_resets(make_environment, length, native):
    transport = CpuTransport([{"rewards": [1., 2., 3.]}] * length, native=native)
    environment = make_environment(transport)
    actions = torch.arange(3 * length).reshape(3, length, 1)
    observations, rewards, terminated, truncated, infos = environment.chunk_step(actions.tolist())
    torch.testing.assert_close(torch.stack(transport.actions, dim=1), actions.float())
    assert rewards.tolist() == [[1.] * length, [2.] * length, [3.] * length]
    assert not terminated.any() and not truncated.any()
    assert len(infos) == len(observations) == length
    assert transport.reset_calls == [[0, 1, 2]]
    assert environment._reset_metrics.call_count == 0
    for index, observation in enumerate(observations):
        assert observation["states"].flatten().tolist() == [index + 1.] * 3


def test_chunk_preserves_first_end_even_when_native_resets_again(make_environment):
    transport = CpuTransport([
        {"rewards": [1., 2., 3.], "truncated": [True, False, False]},
        {"rewards": [1., 2., 3.], "terminated": [False, True, False], "success": [False, True, False]},
        {"rewards": [1., 2., 3.], "terminated": [True, False, False],
         "truncated": [False, False, True], "success": [True, False, False]},
        {"rewards": [4., 5., 6.]},
    ])
    environment = make_environment(transport)
    observations, rewards, terminated, truncated, infos = environment.chunk_step(torch.zeros(3, 3, 1))
    assert rewards.tolist() == [[1., 0., 0.], [2., 2., 0.], [3., 3., 3.]]
    assert terminated[:, -1].tolist() == [False, True, False]
    assert truncated[:, -1].tolist() == [True, False, True]
    assert infos[1]["_final_observation"].tolist() == [False, True, False]
    assert infos[-1]["final_observation"]["states"].flatten().tolist() == [1., 2., 3.]
    assert infos[-1]["final_info"]["episode"]["return"].tolist() == [1., 4., 9.]
    assert infos[-1]["final_info"]["episode"]["episode_len"].tolist() == [1, 2, 3]
    assert infos[-1]["final_info"]["metrics"]["success"].tolist() == [False, True, False]
    assert transport.native_reset_calls == [[0], [1], [0, 2]]
    assert transport.reset_calls == [[0, 1, 2], [1]]
    assert observations[-1]["states"].flatten().tolist() == [0., 0., 0.]
    assert environment.elapsed_steps.tolist() == [0, 0, 0]
    assert environment.returns.tolist() == [0., 0., 0.]
    advance(environment, "chunk")
    transport.states.fill_(99)
    transport.terminal_buffer.fill_(99)
    assert environment.returns.tolist() == [4., 5., 6.]
    assert infos[-1]["final_observation"]["states"].flatten().tolist() == [1., 2., 3.]
    assert infos[-1]["final_info"]["episode"]["return"].tolist() == [1., 4., 9.]


@pytest.mark.parametrize("actions", [
    None, "invalid", [], [[[0.]], []], torch.zeros(3, 1), torch.zeros(3, 0, 1),
    torch.zeros(2, 3, 1), torch.zeros(3, 2, 2), torch.zeros(3, 2, 1, 1),
    torch.full((3, 2, 1), float("nan")), torch.full((3, 2, 1), float("inf")),
])
def test_invalid_chunks_fail_before_simulation_step(make_environment, actions):
    transport = CpuTransport([])
    environment = make_environment(transport)
    with pytest.raises(ValueError, match="shape"):
        environment.chunk_step(actions)
    assert not transport.actions


def test_failed_chunk_clears_context_for_subsequent_steps(make_environment):
    transport = CpuTransport([PARTIAL_END, {"rewards": [4., 5., 6.]}], terminal=False)
    environment = make_environment(transport)
    with pytest.raises(RuntimeError, match="terminal_observation"):
        environment.chunk_step(torch.zeros(3, 3, 1))
    assert environment._chunk_state is None
    environment.reset()
    _, rewards, _, _, _ = advance(environment, "step")
    assert rewards.tolist() == [4., 5., 6.]


@pytest.mark.parametrize("mode", ["step", "step_no_reset", "chunk"])
def test_native_partial_termination_and_timeout_reset_metrics_once(make_environment, mode):
    transport = CpuTransport([PARTIAL_END, {"rewards": [4., 5., 6.]}])
    environment = make_environment(transport)
    observation, rewards, terminated, truncated, info = advance(environment, mode)

    assert observation["states"].flatten().tolist() == [0., 0., 1.]
    assert rewards.tolist() == [1., 2., 3.]
    assert terminated.tolist() == [True, False, False]
    assert truncated.tolist() == [False, True, False]
    assert info["final_observation"]["states"].flatten().tolist() == [1., 1., 1.]
    for name in ("_final_observation", "_final_info", "_elapsed_steps"):
        assert info[name].tolist() == [True, True, False]
    final_episode = info["final_info"]["episode"]
    assert final_episode["return"].tolist() == [1., 2., 3.]
    assert final_episode["episode_len"].tolist() == [1, 1, 1]
    assert final_episode["reward"].tolist() == [1., 2., 3.]
    assert final_episode["success_once"].tolist() == [True, False, False]
    assert final_episode["fail_once"].tolist() == [False, True, False]
    assert environment.returns.tolist() == [0., 0., 3.]
    assert environment.elapsed_steps.tolist() == [0, 0, 1]
    assert not environment.success_once.any() and not environment.fail_once.any()
    assert environment._reset_metrics.call_count == 1
    assert environment._reset_metrics.call_args.args[0].tolist() == [0, 1]
    assert transport.reset_calls == [[0, 1, 2]]
    assert transport.native_reset_calls == [[0, 1]]

    next_obs, _, _, _, next_info = advance(environment, mode)
    assert next_obs["states"].flatten().tolist() == [1., 1., 2.]
    assert "final_observation" not in next_info
    assert environment.returns.tolist() == [4., 5., 9.]
    assert environment.elapsed_steps.tolist() == [1, 1, 2]
    assert environment._reset_metrics.call_count == 1
    # Later simulator mutations cannot change previously emitted rollout data.
    transport.terminal_buffer.fill_(99)
    transport.states.fill_(99)
    environment.returns.fill_(99)
    environment.success_once.fill_(True)
    assert observation["states"].flatten().tolist() == [0., 0., 1.]
    assert info["final_observation"]["states"].flatten().tolist() == [1., 1., 1.]
    assert final_episode["return"].tolist() == [1., 2., 3.]
    assert final_episode["success_once"].tolist() == [True, False, False]


@pytest.mark.parametrize("mode", ["step", "chunk"])
def test_native_matching_horizon_keeps_timeout_bootstrap_state(make_environment, mode):
    transport = CpuTransport([
        {"rewards": [1., 2., 3.]},
        {"rewards": [2., 3., 4.], "truncated": [True] * 3},
    ])
    environment = make_environment(transport, horizon=2)
    advance(environment, mode)
    observation, _, terminated, truncated, info = advance(environment, mode)
    assert not terminated.any() and truncated.all()
    assert not observation["states"].any()
    assert info["final_observation"]["states"].flatten().tolist() == [2., 2., 2.]
    assert info["final_info"]["episode"]["return"].tolist() == [3., 5., 7.]
    assert info["final_info"]["episode"]["episode_len"].tolist() == [2, 2, 2]
    assert transport.reset_calls == [[0, 1, 2]]
    assert environment._reset_metrics.call_count == 1


@pytest.mark.parametrize("mode", ["step", "chunk"])
@pytest.mark.parametrize("frame", [{"rewards": [1.] * 3}, PARTIAL_END])
def test_native_external_horizon_mismatch_fails_without_second_reset(make_environment, mode, frame):
    transport = CpuTransport([frame])
    environment = make_environment(transport, horizon=1)
    with pytest.raises(RuntimeError, match="before native autoreset"):
        advance(environment, mode)
    assert transport.reset_calls == [[0, 1, 2]]
    environment._reset_metrics.assert_not_called()


def test_native_missing_terminal_observation_fails(make_environment):
    transport = CpuTransport([PARTIAL_END], terminal=False)
    environment = make_environment(transport)
    with pytest.raises(RuntimeError, match="requires terminal_observation"):
        environment.step(torch.zeros(3, 1))
    assert transport.reset_calls == [[0, 1, 2]]


@pytest.mark.parametrize("terminal", [
    {"other": torch.zeros(3, 1)}, {"states": torch.zeros(3, 2)},
    {"states": torch.zeros(1, 1)},
])
def test_native_terminal_observation_requires_full_matching_schema(make_environment, terminal):
    transport = CpuTransport([PARTIAL_END])
    native_step = transport.step

    def step(actions):
        result = native_step(actions)
        result[-1]["terminal_observation"] = terminal
        return result

    transport.step = step
    environment = make_environment(transport)
    with pytest.raises(ValueError, match="schema|leading dimension"):
        environment.step(torch.zeros(3, 1))


@pytest.mark.parametrize("mode", ["step", "chunk"])
def test_entity_path_still_lets_rlinf_reset_and_preserve_terminal_state(make_environment, mode):
    transport = CpuTransport([PARTIAL_END], native=False)
    environment = make_environment(transport)
    observation, _, _, _, info = advance(environment, mode)
    assert observation["states"].flatten().tolist() == [0., 0., 1.]
    assert info["final_observation"]["states"].flatten().tolist() == [1., 1., 1.]
    assert transport.reset_calls == [[0, 1, 2], [0, 1]]
    assert environment.elapsed_steps.tolist() == [0, 0, 1]
    assert environment.returns.tolist() == [0., 0., 3.]
    assert environment._reset_metrics.call_count == 1


def test_entity_path_still_accepts_rlinf_owned_timeout(make_environment):
    transport = CpuTransport([{"rewards": [1.] * 3}], native=False)
    environment = make_environment(transport, horizon=1)
    observation, _, _, truncated, info = environment.step(torch.zeros(3, 1))
    assert truncated.all() and not observation["states"].any()
    assert info["final_observation"]["states"].flatten().tolist() == [1., 1., 1.]
    assert transport.reset_calls == [[0, 1, 2], [0, 1, 2]]


def test_native_row_metrics_accumulate_until_that_rows_own_episode_ends(make_environment):
    transport = CpuTransport([
        {"rewards": [1., 2., 3.], "success": [False, False, True]},
        PARTIAL_END,
        {"rewards": [4., 5., 6.], "terminated": [False, False, True]},
    ])
    environment = make_environment(transport)
    advance(environment, "chunk")
    _, _, _, _, early_info = advance(environment, "chunk")
    assert environment.elapsed_steps.tolist() == [0, 0, 2]
    assert environment.returns.tolist() == [0., 0., 6.]
    assert environment.success_once.tolist() == [False, False, True]
    _, _, _, _, last_info = advance(environment, "chunk")
    assert last_info["final_info"]["episode"]["return"].tolist() == [4., 5., 12.]
    assert last_info["final_info"]["episode"]["episode_len"].tolist() == [1, 1, 3]
    assert last_info["final_info"]["episode"]["success_once"].tolist() == [False, False, True]
    assert environment.elapsed_steps.tolist() == [1, 1, 0]
    assert environment.returns.tolist() == [4., 5., 0.]
    assert not environment.success_once.any()
    assert early_info["final_info"]["episode"]["return"].tolist() == [2., 4., 6.]
    assert transport.reset_calls == [[0, 1, 2]]
    assert transport.native_reset_calls == [[0, 1], [2]]
    assert [call.args[0].tolist() for call in environment._reset_metrics.call_args_list] == [[0, 1], [2]]


def test_named_observation_transport_preserves_partial_terminal_rows(make_environment):
    class NamedTransport(CpuTransport):
        def named(self, obs):
            value = obs['states']
            return {'robot.position': value, 'camera.rgb': value[:,None,None,:].expand(-1,4,5,3).to(torch.uint8)}
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            return self.named(obs), info
        def step(self, actions):
            obs, reward, terminated, truncated, info = super().step(actions)
            if 'terminal_observation' in info:
                info['terminal_observation'] = self.named(info['terminal_observation'])
            return self.named(obs), reward, terminated, truncated, info
    transport = NamedTransport([
        {'rewards': [1.,2.,3.], 'terminated': [True,False,False]},
        {'rewards': [1.,2.,3.], 'truncated': [False,True,False]}])
    env = make_environment(transport)
    obs, _ = env.reset()
    assert obs['states']['camera.rgb'].shape == (3,4,5,3)
    obs_list, reward, terminated, truncated, infos = env.chunk_step(torch.zeros(3,2,1))
    terminal = infos[-1]['final_observation']['states']
    assert terminal['robot.position'][:2,0].tolist() == [1.,2.]
    assert terminal['camera.rgb'].dtype == torch.uint8
    assert transport.native_reset_calls == [[0],[1]]
    # Row 0 advanced in the next episode during the second chunk slot; restore
    # its clean reset at the chunk boundary without losing its first terminal.
    assert transport.reset_calls == [[0,1,2],[0,1,2],[0]]
