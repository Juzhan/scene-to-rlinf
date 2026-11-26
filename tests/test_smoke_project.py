"""Object projects smoke-test through the shared registry and the project model: no trainer needed."""

import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from shared.assembly import project as project_module
from tools.smoke import smoke_project


class FakeEnvironment:
    device = torch.device("cpu")
    num_envs = 2

    def __init__(self):
        self.actions = []

    def reset(self, env_ids=None, seed=None):
        return {"robot.joint_position": torch.zeros(2, 3)}, {}

    def step(self, actions):
        self.actions.append(actions)
        observations = {"robot.joint_position": actions.detach().clone()}
        rewards = torch.zeros(2)
        terminated = torch.zeros(2, dtype=torch.bool)
        truncated = torch.zeros(2, dtype=torch.bool)
        if len(self.actions) == 2:
            truncated[:] = True  # one reset inside the run, so the loop covers the second step too
        return observations, rewards, terminated, truncated, {}

    def close(self):
        self.closed = True


def test_smoke_project_steps_the_project_model_without_a_trainer(tmp_path, monkeypatch):
    environment = FakeEnvironment()
    calls = []

    class Model(torch.nn.Module):
        def forward(self, obs):
            calls.append(obs.robot.joint_position.shape)
            return obs.robot.joint_position + 1

    class Obs:
        def __init__(self, values):
            self.robot = SimpleNamespace(joint_position=values["robot.joint_position"])

    project = SimpleNamespace(model_builder=lambda: (lambda config, dtype: Model()))
    module = SimpleNamespace(project=project, __name__="fake_project", __file__=str(tmp_path / "project.py"))
    monkeypatch.setitem(__import__("sys").modules, "fake_project.observation", SimpleNamespace(Obs=Obs))
    monkeypatch.setattr(project_module, "environment_factory", lambda task_id: (
        lambda parameters, count, seed, horizon: (environment, None)))
    cfg = OmegaConf.create({"env": {"train": {"total_num_envs": 4, "seed": 0, "max_episode_steps": 8,
                                              "init_params": {"id": "fake-task", "device": "cpu"}}},
                            "actor": {"model": {}},
                            "runner": {"logger": {"log_path": str(tmp_path)}}})

    report = smoke_project(cfg, module, steps=3)

    assert report["num_envs"] == 4  # capped at 4 batch rows
    assert report["steps"] == 3 and report["contract_passed"] and report["clean_exit"]
    assert report["resets"] == 2  # both rows truncate on the second step
    assert calls == [(2, 3)] * 3
    assert len(environment.actions) == 3
    torch.testing.assert_close(environment.actions[0], torch.ones(2, 3))
    saved = json.loads(next(tmp_path.glob("checks/*.json")).read_text())
    assert saved["task"] == "fake-task" and saved["action_shape"] == [2, 3]


pytestmark = pytest.mark.group_isaaclab3  # runs in the [isaaclab3] dependency group
