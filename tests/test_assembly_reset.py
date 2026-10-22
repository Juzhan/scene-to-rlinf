"""Reset sequencing and replica isolation without starting Isaac or a GPU."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from shared.assembly import Entity, Pose, RootReset, Scene, WorldSpec
from shared.backends.isaac.preview_world import IsaacAssemblyWorld
from shared.backends.isaac.preview_runtime import RuntimeWorld


class MemoryRigidObject:
    """Selected-index writes model the contract of native rigid objects."""

    def __init__(self, num_envs):
        defaults = torch.zeros(num_envs, 13)
        defaults[:, 3] = 1
        self.data = SimpleNamespace(default_root_state=defaults)
        self.state = defaults.clone()
        self.state[:, :3] = 20
        self.randomized_ids = []

    def write_root_state_to_sim(self, state, *, env_ids):
        self.state[env_ids] = state

    def write_root_pose_to_sim(self, pose, *, env_ids):
        self.state[env_ids, :7] = pose
        self.randomized_ids.append(env_ids.tolist())


@pytest.fixture
def make_world(monkeypatch):
    # reset imports these APIs after AppLauncher in production. No physics is
    # needed to check ordering of writes, random sampling and replica indices.
    class NativeArticulation:
        pass

    monkeypatch.setitem(sys.modules, "isaaclab.assets", SimpleNamespace(Articulation=NativeArticulation))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", SimpleNamespace(
        combine_frame_transforms=Mock(), subtract_frame_transforms=Mock()))
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", SimpleNamespace(
        reset_root_state_uniform=Mock(side_effect=AssertionError("Discrete resets must not use range sampling"))))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(SceneEntityCfg=Mock()))
    simulation = SimpleNamespace(forward=Mock())
    monkeypatch.setitem(sys.modules, "isaaclab.sim", SimpleNamespace(
        SimulationContext=SimpleNamespace(instance=lambda: simulation)))

    def build(*, extra_assets=0):
        # Both random objects have a range of discrete states. Calling the real
        # sampler reveals repeated writes and RNG consumption, not just counts
        # on a replacement for _randomize_root_poses.
        rule = RootReset(states=tuple(Pose((i / 10, 0, 0.5)) for i in range(10)))
        entities = [Entity("cube", factory="box", reset=rule, initial_state={"active": False}),
                    Entity("target", factory="box", reset=rule)]
        entities.extend(Entity(f"other_{i}", factory="box") for i in range(extra_assets))
        world = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
        RuntimeWorld.__init__(world, WorldSpec(Scene("world", children=tuple(entities))),
                              num_envs=3, step_dt=0.01)
        world.device = "cpu"
        world.keys = {"world/" + node.name: node.name for node in entities}
        objects = {node.name: MemoryRigidObject(3) for node in entities}
        world.scene = SimpleNamespace(
            articulations={}, rigid_objects=objects,
            env_origins=torch.tensor([[0., 0, 0], [4., 0, 0], [8., 0, 0]]),
            update=Mock(), reset=Mock(), write_data_to_sim=Mock())
        world.asset = lambda path: objects[world.keys[path]]
        world.connections, world.mounts = (), {}
        world.elapsed_steps = torch.tensor([4, 5, 6])
        world._sensor_epoch = 0
        world.seed(42)
        return world

    return build


def test_reset_samples_each_random_asset_once_after_default_restoration(make_world):
    world = make_world(extra_assets=2)
    world.reset()
    for name in ("cube", "target"):
        asset = world.scene.rigid_objects[name]
        assert asset.randomized_ids == [[0, 1, 2]]
        # A subsequent default-state write must not erase sampled poses.
        torch.testing.assert_close(asset.state[:, 2], torch.full((3,), 0.5))
    assert world.elapsed_steps.tolist() == [0, 0, 0]


def test_partial_reset_preserves_other_replicas_and_empty_reset_is_noop(make_world):
    world = make_world(extra_assets=1)
    world.set_state("world/cube", "active", True)
    before = {name: asset.state.clone() for name, asset in world.scene.rigid_objects.items()}
    world.reset([1])
    for name, asset in world.scene.rigid_objects.items():
        torch.testing.assert_close(asset.state[[0, 2]], before[name][[0, 2]])
    for name in ("cube", "target"):
        asset = world.scene.rigid_objects[name]
        assert asset.randomized_ids == [[1]]
        assert 4 <= asset.state[1, 0] <= 4.9  # Local sample plus replica origin.
        assert asset.state[1, 2] == 0.5
    assert world.semantic_state["world/cube"] == [{"active": True}, {"active": False}, {"active": True}]
    assert world.elapsed_steps.tolist() == [4, 0, 6]
    assert world.scene.reset.call_args.args[0].tolist() == [1]
    rng_state, epoch = world.rng.getstate(), world._sensor_epoch
    world.reset([])
    assert world.rng.getstate() == rng_state and world._sensor_epoch == epoch
    assert world.scene.reset.call_count == 1


def test_unrelated_assets_do_not_change_reset_samples(make_world):
    original, expanded = make_world(), make_world(extra_assets=3)
    original.reset([0, 2])
    expanded.reset([0, 2])
    for name in ("cube", "target"):
        torch.testing.assert_close(original.scene.rigid_objects[name].state,
                                   expanded.scene.rigid_objects[name].state)
    assert original.rng.getstate() == expanded.rng.getstate()
