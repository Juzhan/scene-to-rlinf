from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

from shared.backends.isaac.environment import Environment
from shared.integrations.rlinf.isaac import _ENVIRONMENT_FACTORIES
from tools.go2_evaluation import evaluation_step_state, physical_evaluation_state
from tools.preview import parse_options, preview


def test_preview_leaves_episode_resets_to_the_native_environment(tmp_path):
    options, _ = parse_options(["--steps", "2", "--num-envs", "2"], "view")
    observations = {"states": torch.zeros(2, 4)}
    native = SimpleNamespace(
        device="cpu", step_dt=0.001, action_manager=SimpleNamespace(total_action_dim=1),
        sim=SimpleNamespace(is_stopped=Mock(return_value=False), is_playing=Mock(return_value=True)),
    )
    environment = object.__new__(Environment)
    environment.native = native
    environment.device = torch.device("cpu")
    environment.reset = Mock(return_value=(observations, {}))
    environment.close = Mock()
    environment.step = Mock(return_value=(
        observations, torch.ones(2), torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool),
        {"autoreset": True},
    ))
    owner = SimpleNamespace(application=SimpleNamespace(is_running=Mock(return_value=True)))
    cfg = OmegaConf.create({
        "env": {"train": {"env_type": "isaaclab", "seed": 42, "max_episode_steps": 8,
                          "init_params": {"id": "test-consumer", "obs_dim": 4, "action_dim": 1}}},
        "runner": {"logger": {"log_path": str(tmp_path)}},
    })
    window_module = SimpleNamespace(get_default_app_window=Mock())
    constructor = Mock(return_value=(environment, owner))
    with patch.dict(_ENVIRONMENT_FACTORIES, {"test-consumer": constructor}), \
         patch.dict("os.environ", {"DISPLAY": ":test"}), \
         patch.dict("sys.modules", {"omni.appwindow": window_module}), patch("builtins.print"):
        report = preview(cfg, options, recording=False)
    assert environment.reset.call_count == 1
    assert report["resets"] == 2
    assert report["status"] == "completed"
    environment.close.assert_called_once()


def test_evaluation_uses_terminal_physics_for_done_rows_without_changing_live_state():
    robot = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.3], [1.0, 0.0, 0.4], [0.0, 0.0, 0.3]]),
        heading_w=torch.tensor([0.0, 0.1, 0.0]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
        root_lin_vel_w=torch.zeros(3, 3),
    ))
    scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=torch.zeros(3, 2, 3)))
    live = physical_evaluation_state(robot, scanner)
    terminal = {name: value.clone() for name, value in live.items()}
    # The first row fell; the last row timed out at the goal. Both were reset.
    terminal["root_pos_w"][:] = torch.tensor([[2.0, 0.0, 0.1], [99.0, 99.0, 99.0], [3.0, 0.0, 1.3]])
    terminal["projected_gravity_b"][0, 2] = 0.0
    terminal["heading_w"][0] = 0.6
    terminal["root_lin_vel_w"][0, 0] = 0.7
    terminal["ray_hits_w"][2, :, 2] = 1.0
    original = {name: value.clone() for name, value in live.items()}
    result = evaluation_step_state(live, torch.tensor([True, False, True]), {
        "autoreset": True, "terminal_state": terminal,
    })
    for name in result:
        torch.testing.assert_close(result[name][[0, 2]], terminal[name][[0, 2]])
        torch.testing.assert_close(result[name][1], live[name][1])
        torch.testing.assert_close(live[name], original[name])
    # Progress, clearance, posture, heading and speed refer to the final step.
    torch.testing.assert_close(result["root_pos_w"][:, 0], torch.tensor([2.0, 1.0, 3.0]))
    assert result["root_pos_w"][2, 2] - result["ray_hits_w"][2, 0, 2] == pytest.approx(0.3)
    assert result["projected_gravity_b"][0, 2] == 0
    assert result["heading_w"][0] == pytest.approx(0.6)
    assert result["root_lin_vel_w"][0, 0] == pytest.approx(0.7)


def test_evaluation_requires_snapshot_only_when_native_step_reset_rows():
    live = {"root_pos_w": torch.zeros(2, 3)}
    assert evaluation_step_state(live, torch.tensor([False, False]), {"autoreset": True}) is live
    assert evaluation_step_state(live, torch.tensor([True, False]), {}) is live
    with pytest.raises(RuntimeError, match="terminal physical state"):
        evaluation_step_state(live, torch.tensor([True, False]), {"autoreset": True})
    with pytest.raises(ValueError, match="full environment batch"):
        evaluation_step_state(live, torch.tensor([True, False]), {
            "autoreset": True, "terminal_state": {"root_pos_w": torch.ones(1, 3)},
        })
