"""Abstract scene entities: geometry, mounts, reset sampling and native wiring."""

import json
from math import pi, sqrt
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from shared.assembly.abstract import (AbstractState, abstract_catalog, abstract_entities, commanded, commanded_entity,
                                      mounted_pose, reset_areas, sample_poses, sample_positions, tensor_field)
from shared.assembly.layout import world_spec_from_config
from shared.assembly.readers import bind_common
from shared.assembly.spec import Abstract, RootReset, Scene, WorldSpec
from shared.backends.isaac import abstract_terms
from shared.backends.isaac.abstract_terms import reset_root_in_regions
from shared.backends.isaac.scene import event_configuration


def scene(*entities, **extra):
    return {"schema_version": 1, "name": "world", "entities": list(entities), **extra}


def area(name="area", size=(1., 2.), **fields):
    return {"name": name, "kind": "abstract", "shape": "rectangle", "size": list(size), **fields}


def world(*entities, tmp_path):
    return world_spec_from_config(scene(*entities), directory=tmp_path)


@pytest.mark.parametrize("entry", [
    {"name": "a", "kind": "abstract", "shape": "circle"},
    {"name": "a", "kind": "abstract", "shape": "box", "size": [1, 1]},
    {"name": "a", "kind": "abstract", "shape": "rectangle", "size": [1, float("inf")]},
    {"name": "a", "kind": "abstract", "shape": "rectangle", "size": [1, 0]},
    {"name": "a", "kind": "abstract", "shape": "point", "size": [1]},
    {"name": "a", "kind": "abstract"},
    {"name": "a", "kind": "abstract", "shape": "point", "factory": "dex_cube"},
    {"name": "a", "kind": "abstract", "shape": "point", "children": []},
    {"name": "a", "factory": "dex_cube", "shape": "point"},
])
def test_abstract_rejects_ambiguous_geometry_and_physical_fields(entry, tmp_path):
    with pytest.raises(ValueError):
        world(entry, tmp_path=tmp_path)


def test_regions_section_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="abstract"):
        world_spec_from_config(scene(regions={"a": {"shape": "point"}}), directory=tmp_path)


def test_reset_region_must_be_a_static_abstract_area(tmp_path):
    cube = {"name": "cube", "factory": "dex_cube", "reset": {"region": "world/area"}}
    with pytest.raises(ValueError, match="rectangle or box"):
        world(cube, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="rectangle or box"):
        world(cube, {"name": "area", "kind": "abstract", "shape": "point"}, tmp_path=tmp_path)
    world(cube, area(), tmp_path=tmp_path)
    with pytest.raises(ValueError, match="static"):
        world(cube, area(mount={"entity": "world/robot"}), {"name": "robot", "factory": "dex_cube"},
              tmp_path=tmp_path)
    with pytest.raises(ValueError, match="consumes"):
        world({**cube, "reset": {"region": "world/area", "inset": .5}}, area(), tmp_path=tmp_path)
    with pytest.raises(ValueError, match="only available on abstract"):
        world({**cube, "reset": {"region": "world/area", "interval": 2.}}, area(), tmp_path=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        RootReset(region="world/area", pose_range={"x": [0, 1]})


def test_abstract_mount_follows_an_entity_root_and_cannot_be_randomized(tmp_path):
    robot = {"name": "robot", "factory": "dex_cube"}
    point = {"name": "tip", "kind": "abstract", "shape": "point", "mount": {"entity": "world/robot"}}
    world(robot, point, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="body mounts"):
        world(robot, {**point, "mount": {"entity": "world/robot", "body": "base"}}, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="cannot be randomized"):
        world(robot, area(), {**point, "reset": {"region": "world/area"}}, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="static"):
        world(area(reset={"region": "world/area"}), tmp_path=tmp_path)


def test_scene_has_no_motion_field_the_task_moves_entities(tmp_path):
    robot = {"name": "robot", "factory": "dex_cube"}
    ball = {"name": "ball", "kind": "abstract", "shape": "point"}
    world(robot, ball, tmp_path=tmp_path)
    with pytest.raises(ValueError, match="motion"):
        world(robot, {**ball, "motion": ".task:move_ball"}, tmp_path=tmp_path)


def test_unmounted_non_region_entities_are_commanded_with_linear_velocity():
    settings = scene(area(), {"name": "ball", "kind": "abstract", "shape": "point"},
                     {"name": "robot", "factory": "dex_cube"},
                     {"name": "tip", "kind": "abstract", "shape": "point", "mount": {"entity": "world/robot"}},
                     {"name": "goal", "kind": "abstract", "shape": "point", "reset": {"region": "world/area"}})
    assert set(commanded(settings)) == {"ball", "goal"}
    catalog = abstract_catalog(settings)
    value, information = catalog["ball.linear_velocity"]
    assert value["commanded"] and value["field"] == "linear_velocity"
    assert information.unit == "m/s" and information.frame == "environment"
    assert catalog["goal.linear_velocity"][0]["commanded"]
    # Regions stay static constants; mounted entities follow their entity.
    assert "area.linear_velocity" not in catalog and catalog["area.position"][0] == [0., 0., 0.]
    assert "tip.linear_velocity" not in catalog and not catalog["tip.position"][0]["commanded"]
    entity = commanded_entity(settings, abstract_entities(settings)["ball"])
    assert entity["name"] == "ball" and entity["reset"] is None and entity["mount"] is None


def _state(reset=None, envs=3):
    entity = {"name": "ball", "shape": "point", "size": [], "mount": None, "reset": reset,
              "pose": {"position": [1., 0., .2], "orientation": [1., 0., 0., 0.]}}
    areas = {"world/area": {"shape": "rectangle", "size": [1., 1.],
                            "pose": {"position": [5., 5., 0.], "orientation": [1., 0., 0., 0.]}}}
    return AbstractState(entity, areas, envs, "cpu")


def test_abstract_state_resamples_selected_rows_and_clears_their_velocity():
    state = _state(reset={"region": "world/area", "yaw_range": [0., 0.]})
    state.velocity[:] = 1.
    state.fresh[:] = False
    state.resample(torch.tensor([1]))
    assert state.velocity[1].abs().sum() == 0 and state.velocity[[0, 2]].eq(1.).all()
    assert state.fresh.tolist() == [False, True, False]
    assert (state.pose[1, :2] - 5.).abs().max() <= .5 and state.pose[[0, 2], 0].eq(1.).all()


def test_abstract_state_applies_task_sets_to_rows_not_freshly_reset():
    state = _state()
    state.resample(torch.tensor([0, 1, 2]))
    state.advance(.5, torch.tensor([[2., 0., .2]] * 3))
    torch.testing.assert_close(state.pose[:, 0], torch.ones(3))  # first update after reset keeps the pose
    assert state.velocity.abs().sum() == 0
    state.advance(.5, torch.tensor([[2., 0., .2]] * 3))
    state.resample(torch.tensor([2]))
    pose = torch.tensor([[3., 0., .2, 0., 0., 0., 2.]] * 3)  # xyz + unnormalized wxyz
    state.advance(.5, pose)
    torch.testing.assert_close(state.pose[:, 0], torch.tensor([3., 3., 1.]))
    torch.testing.assert_close(state.read("linear_velocity"), torch.tensor([[2., 0., 0.]] * 2 + [[0., 0., 0.]]))
    torch.testing.assert_close(state.read("orientation"), torch.tensor([[0., 0., 0., 1.]] * 2 + [[1., 0., 0., 0.]]))
    state.advance(.5)  # not set this step: stays, velocity drops to 0
    assert state.velocity.abs().sum() == 0 and state.pose[0, 0] == 3.
    state.read("position")[:] = 0.  # readers get copies
    assert state.pose[0, 0] == 3.


def test_catalog_exposes_constant_mounted_and_sampled_fields():
    settings = scene(
        area(pose={"position": [1, 2, 3]}),
        {"name": "robot", "factory": "dex_cube"},
        {"name": "axis", "kind": "abstract", "shape": "direction", "mount": {"entity": "world/robot"},
         "pose": {"orientation": [sqrt(.5), 0, 0, sqrt(.5)]}},
        {"name": "goal", "kind": "abstract", "shape": "point",
         "reset": {"region": "world/area", "interval": 4}})
    catalog = abstract_catalog(settings)
    assert catalog["area.position"][0] == [1., 2., 3.]
    assert catalog["area.size"][0] == [1., 2.]
    assert "area.direction" not in catalog and "goal.size" not in catalog
    axis = catalog["axis.direction"][0]
    assert axis["mount"] == "robot" and not axis["commanded"]
    assert catalog["goal.position"][0]["commanded"]
    assert catalog["axis.orientation"][1].frame == "environment"
    assert reset_areas(settings) == {"world/area": {
        "shape": "rectangle", "size": [1., 2.],
        "pose": {"position": [1., 2., 3.], "orientation": [1., 0., 0., 0.]}}}


def test_mounted_reading_composes_root_pose_per_environment():
    settings = scene(
        {"name": "robot", "factory": "dex_cube"},
        {"name": "axis", "kind": "abstract", "shape": "direction", "mount": {"entity": "world/robot"},
         "pose": {"position": [1., 0., 0.]}})
    # Environment 1 is yawed 90 degrees: its local +X offset and direction become +Y.
    position = torch.tensor([[0., 0., 0.], [5., 0., 1.]])
    orientation = torch.tensor([[1., 0., 0., 0.], [sqrt(.5), 0., 0., sqrt(.5)]])
    roots = {"robot.position": position, "robot.orientation": orientation}
    env = SimpleNamespace(device="cpu", num_envs=2)
    read = lambda path: bind_common(env, settings, path, bind_physical=lambda name: lambda: roots[name],
                                    bind_sampled_target=None)()
    torch.testing.assert_close(read("axis.position"), torch.tensor([[1., 0., 0.], [5., 1., 1.]]))
    torch.testing.assert_close(read("axis.direction"), torch.tensor([[1., 0., 0.], [0., 1., 0.]]))
    torch.testing.assert_close(read("axis.orientation"), orientation)


def test_rotated_rectangle_sampling_is_seeded_and_stays_in_its_plane():
    # 90-degree rotation about world Y turns local rectangle X into world -Z.
    region = {"shape": "rectangle", "size": [4, 2],
              "pose": {"position": [10, 20, 30], "orientation": [sqrt(.5), 0, sqrt(.5), 0]}}
    first = sample_positions(region, 3000, device="cpu", inset=.2, generator=torch.Generator().manual_seed(72))
    repeated = sample_positions(region, 3000, device="cpu", inset=.2, generator=torch.Generator().manual_seed(72))
    torch.testing.assert_close(first, repeated)
    assert torch.max(torch.abs(first[:, 0] - 10)) < 1e-5
    assert torch.max(torch.abs(first[:, 1] - 20)) <= .8
    assert torch.max(torch.abs(first[:, 2] - 30)) <= 1.8
    torch.testing.assert_close(first.mean(0), torch.tensor([10., 20., 30.]), atol=.05, rtol=0)


def test_box_sampling_has_three_dimensions_and_rejects_empty_inset():
    region = {"shape": "box", "size": [2, 4, 6]}
    points = sample_positions(region, 1000, device="cpu", generator=torch.Generator().manual_seed(2))
    assert torch.all(points.abs() <= torch.tensor([1., 2., 3.]))
    assert torch.all(points.std(0) > .5)
    with pytest.raises(ValueError, match="nonempty"):
        sample_positions(region, 1, device="cpu", inset=1.)


def test_sampled_pose_heading_includes_area_rotation():
    areas = {"world/area": {"shape": "rectangle", "size": [1., 1.],
                            "pose": {"position": [0., 0., 0.], "orientation": [sqrt(.5), 0, 0, sqrt(.5)]}}}
    base = {"position": [0., 0., 0.], "orientation": [1., 0., 0., 0.]}
    pose = sample_poses({"region": "world/area", "yaw_range": [.3, .3]}, base, areas, 5, device="cpu")
    heading = torch.atan2(tensor_field(pose, "direction")[:, 1], tensor_field(pose, "direction")[:, 0])
    torch.testing.assert_close(heading, torch.full((5,), pi / 2 + .3))


def test_region_reset_preserves_other_environments_and_joint_state():
    defaults = torch.zeros((4, 13))
    defaults[:, 3] = 1
    defaults[:, 7:] = 4
    written = torch.arange(52, dtype=torch.float).reshape(4, 13)
    before = written.clone()

    class Body:
        data = SimpleNamespace(default_root_state=defaults, joint_pos=torch.tensor([[1., 2.]]).repeat(4, 1))

        def write_root_state_to_sim(self, states, *, env_ids):
            written[env_ids] = states

    class NativeScene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [10., 0., 2.], [20., 0., 0.], [30., 0., 0.]])

    env = SimpleNamespace(num_envs=4, device="cpu", scene=NativeScene(box=Body()))
    rule = {"region": "world/area", "yaw_range": [.3, .3], "inset": .1}
    areas = {"world/area": {"shape": "rectangle", "size": [2., 4.], "pose": {
        "position": [1., 2., .5], "orientation": [sqrt(.5), 0, 0, sqrt(.5)]}}}
    ids = torch.tensor([1, 3])
    torch.manual_seed(18)
    reset_root_in_regions(env, ids, rules={"box": rule}, areas=areas)
    first = written.clone()
    torch.testing.assert_close(written[[0, 2]], before[[0, 2]])
    torch.testing.assert_close(written[ids, 2], torch.tensor([2.5, .5]))
    assert not written[ids, 7:].any()
    torch.testing.assert_close(written[ids, 3:7].norm(dim=1), torch.ones(2))
    angle = torch.tensor((pi / 2 + .3) / 2)
    torch.testing.assert_close(written[ids, 3], angle.cos().repeat(2))
    torch.testing.assert_close(written[ids, 6], angle.sin().repeat(2))
    torch.manual_seed(18)
    reset_root_in_regions(env, ids, rules={"box": rule}, areas=areas)
    torch.testing.assert_close(written, first)
    reset_root_in_regions(env, [], rules={"box": rule}, areas=areas)
    torch.testing.assert_close(written, first)
    env.scene["box"].is_fixed_base = True
    with pytest.raises(ValueError, match="fixed-base"):
        reset_root_in_regions(env, ids, rules={"box": rule}, areas=areas)


def test_area_reset_compiles_to_one_serializable_native_event(tmp_path, monkeypatch):
    manager = ModuleType("isaaclab.managers")
    manager.EventTermCfg = SimpleNamespace
    monkeypatch.setitem(sys.modules, manager.__name__, manager)
    cube = {"name": "box", "factory": "dex_cube", "reset": {"region": "world/table"}}
    table = area("table", (1., .5), pose={"position": [0, 0, .1]})
    events = event_configuration(world(cube, table, tmp_path=tmp_path))
    assert list(events) == ["assembly_region_poses"]
    event = events["assembly_region_poses"]
    assert event.mode == "reset" and event.func is reset_root_in_regions
    assert event.params["rules"]["box"]["region"] == "world/table"
    assert event.params["areas"]["world/table"]["size"] == [1., .5]
    json.dumps(event.params)
    # Geometry alone, or a randomized abstract entity, adds no asset reset event.
    goal = {"name": "goal", "kind": "abstract", "shape": "point", "reset": {"region": "world/table"}}
    assert event_configuration(world(table, goal, tmp_path=tmp_path)) == {}


def test_command_terms_cover_commanded_and_shown_entities_only(monkeypatch):
    monkeypatch.setattr(abstract_terms, "_config_type", lambda: SimpleNamespace)
    settings = scene(area(), {"name": "goal", "kind": "abstract", "shape": "point",
                              "reset": {"region": "world/area", "interval": 4}},
                     {"name": "still", "kind": "abstract", "shape": "point"},
                     visualization={"abstract": ["area"]})
    output = {"commands": SimpleNamespace()}
    abstract_terms.attach_abstract(output, settings)
    terms = vars(output["commands"])
    # "still" is static but the Task may move it; the region stays a constant (term only to draw it).
    assert set(terms) == {"assembly_abstract_area", "assembly_abstract_goal", "assembly_abstract_still"}
    assert terms["assembly_abstract_still"].resampling_time_range == (1e9, 1e9)
    goal, shown = terms["assembly_abstract_goal"], terms["assembly_abstract_area"]
    assert goal.resampling_time_range == (4., 4.) and not goal.debug_vis
    assert goal.entity["reset"]["region"] == "world/area" and "world/area" in goal.areas
    assert shown.debug_vis and shown.entity["reset"] is None


def test_legacy_preview_rejects_abstract_entities_before_importing_simulator():
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    with pytest.raises(NotImplementedError, match="native Project environment"):
        IsaacAssemblyWorld(WorldSpec(Scene("world", children=(Abstract("a", shape="point"),))))


def test_assembly_override_replaces_reset_mode(tmp_path):
    from omegaconf import OmegaConf

    OmegaConf.save(OmegaConf.create({"schema_version": 1, "root": {
        "name": "cube", "factory": "dex_cube", "reset": {"pose_range": {"x": [-.1, .1]}}}}),
        tmp_path / "cube.yaml")
    cube = {"name": "cube", "assembly": "cube.yaml", "overrides": {".": {"reset": {"region": "world/area"}}}}
    body = world(area(), cube, tmp_path=tmp_path).index()["world/cube"][0]
    assert body.reset.region == "world/area" and not body.reset.pose_range
