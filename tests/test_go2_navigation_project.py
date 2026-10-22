from pathlib import Path

import torch

from projects.go2_navigation.runtime import _load_low_level_actor
from shared.assembly.editing import SceneDocument


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects/go2_navigation"


def test_navigation_scene_is_static_checkable_and_exposes_high_level_contract():
    scene = PROJECT / "scene.yaml"
    snapshot = SceneDocument(scene).snapshot()
    assert list(snapshot["actions"]) == ["navigation_velocity"]
    assert snapshot["observations"]["states"]["terms"] == ["base_lin_vel", "projected_gravity", "target_pose"]
    assert snapshot["task"]["preset"] == "velocity"


def test_low_level_checkpoint_loads_into_our_frozen_network(tmp_path, monkeypatch):
    from projects.go2_navigation import runtime

    actor = runtime._actor_network()
    source = tmp_path / "weights.pt"
    torch.save({"actor." + name: value for name, value in actor.state_dict().items()}, source)
    monkeypatch.setattr(runtime, "DEFAULT_LOW_LEVEL", source)
    policy = _load_low_level_actor(None, "cpu")
    sample = torch.randn(4, 235)
    torch.testing.assert_close(policy(sample), actor.eval()(sample), rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in policy.parameters())
