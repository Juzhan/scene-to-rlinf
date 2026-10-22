"""Composition, standalone preview and training boundaries without a simulator."""

from math import sqrt
from pathlib import Path
import pytest

from shared.assembly import (
    Entity, FixedJoint, Mount, Observation,
    Pose, Robot, Scene, Sensor, Signal, Signals, WorldSpec,
)
from shared.backends.isaac.preview_environment import AssemblyEnv, Evaluation
from projects.assembly_demo.layout import load_assembly, load_world_spec, scene_config_path

ARM_CONFIG = Path(__file__).resolve().parents[1] / "projects/assembly_demo/assemblies/arm.yaml"


def test_go2_evaluation_routes_cover_motion_and_body_directions():
    from tools.go2_evaluation import route_cases, summarize

    cases = route_cases([("narrow_up", 2), ("narrow_down", 6)], [0, 9], [0, 45, 90], [0, 90, 180, 270])
    assert len(cases) == 96
    for case in cases:
        assert max(abs(value) for value in case["start_xy"]) == pytest.approx(1.1 if case["outward"] else 3.4)
        delta = [goal - start for goal, start in zip(case["goal_xy"], case["start_xy"])]
        assert sum(value * unit for value, unit in zip(delta, case["unit_xy"])) > 0
    records = [dict(cases[0], motion="up", success=False, fell=True),
               dict(cases[0], motion="up", success=True, fell=False)]
    summary = summarize(records)
    assert summary["episodes"] == 2
    assert summary["success_rate"] == summary["fall_rate"] == 0.5
    assert summary["by_case"][0]["episodes"] == 2


def test_lift_scene_uses_shared_asset_catalog_and_loader():
    from shared.backends.isaac.assets import FACTORIES
    from shared.assembly.layout import load_world_spec, load_scene_config

    path = ARM_CONFIG.parents[2] / "assembly_lift/scene.yaml"
    source = path.read_text()
    for implementation_detail in ("usd_path", "prim_path", "ISAAC", "spawn:"):
        assert implementation_detail not in source
    description = load_world_spec(path)
    entities = description.index()
    for node, _ in entities.values():
        if node.factory:
            assert node.factory in FACTORIES
    assert entities["world/table"][0].factory == "seattle_table"
    assert entities["world/object"][0].factory == "dex_cube"
    settings = load_scene_config(path)
    # The template declares planner/executor inputs and named control targets
    # instead of native observation groups and action terms.
    assert settings["control"]["train"] == "executor"
    assert set(settings["control"]["output"]) == {"arm", "gripper"}
    assert settings["control"]["output"]["arm"]["joint_names"] == [f"panda_joint{i}" for i in range(1, 8)]
    assert "object.position" in settings["control"]["components"]["executor"]["observations"]
    assert "task" not in settings and "observations" not in settings


def test_entity_backend_copies_callable_asset_factory_results():
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    config = {"nested": {"size": [1, 2, 3]}}
    world = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    world.factories = {"preset": lambda: config}
    node = Entity("object", factory="preset")
    first = world._asset_config(node)
    first["nested"]["size"][0] = 9
    assert world._asset_config(node)["nested"]["size"] == [1, 2, 3]
    assert config["nested"]["size"] == [1, 2, 3]


def test_manager_scene_references_asset_catalog_without_native_resource_paths():
    from omegaconf import OmegaConf
    from shared.backends.isaac.assets import FACTORIES, MATERIALS

    path = ARM_CONFIG.parents[2] / "assembly_go2/scene.yaml"
    source = path.read_text()
    settings = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    for implementation_detail in ("mdl_path", "texture_file", "prim_path", "ISAAC", "_target_"):
        assert implementation_detail not in source
    entities = {entry["name"]: entry for entry in settings["entities"]}
    for component in entities.values():
        assert component["factory"] in FACTORIES
    terrain = entities["terrain"]
    assert terrain["parameters"]["material"] in MATERIALS
    scanner = entities["height_scanner"]
    assert scanner["mount"] == {"entity": "world/robot", "body": "base"}
    assert scanner["meshes"] == ["world/terrain"]
    assert scanner["pose"]["position"] == [0.0, 0.0, 20.0]


def test_root_reset_specs_accept_states_or_ranges():
    from shared.assembly.spec import RootReset

    state = Pose((0.1, 0.0, 0.831))
    assert RootReset(states=(state,)).states[0] == state
    assert RootReset(pose_range={"x": [-0.03, 0.03]}).pose_range["x"] == (-0.03, 0.03)
    for kwargs in ({}, {"states": (state,), "pose_range": {"x": [0, 1]}},
                   {"pose_range": {"x": [1, -1]}}, {"pose_range": {"x": [0, float("nan")]}},
                   {"pose_range": {"invalid": [0, 1]}}, {"states": ({},)}):
        with pytest.raises(ValueError): RootReset(**kwargs)
    with pytest.raises(ValueError):
        WorldSpec(Scene("world", reset=RootReset(pose_range={"x": [0, 1]}))).index()


@pytest.mark.parametrize("mode", ["states", "range"])
def test_scene_loads_entity_reset_rules(tmp_path, mode):
    from omegaconf import OmegaConf

    rule = ({"states": [{"position": [0.1, 0.0, 0.831]}]}
            if mode == "states" else {"pose_range": {"x": [-0.03, 0.03]}})
    config = {"schema_version": 1, "name": "world", "entities": [
        {"name": "cube", "factory": "box", "reset": rule}]}
    path = tmp_path / "scene.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    reset = load_world_spec(path).index()["world/cube"][0].reset
    if mode == "states":
        assert reset.states[0].position == (0.1, 0.0, 0.831)
    else:
        assert reset.pose_range == {"x": (-0.03, 0.03)}
    config["entities"][0]["reset"] = {"pose_range": {"x": [1, -1]}}
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValueError, match="ordered"): load_world_spec(path)
    config["entities"][0]["reset"] = {"velocity_range": {"x": [0, 1]}}
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValueError, match="Unknown reset fields"): load_world_spec(path)


@pytest.mark.parametrize("mode", ["states", "range"])
def test_backend_pose_reset_sampling_seed_and_partial_isolation(monkeypatch, mode):
    import sys
    import torch
    from random import Random
    from types import SimpleNamespace
    from shared.assembly import RootReset
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    written = torch.zeros(4, 7)
    defaults = torch.zeros(4, 13)
    defaults[:, 2], defaults[:, 3] = 0.831, 1
    origins = torch.tensor([[0., 0, 0], [4., 0, 0], [8., 0, 0], [12., 0, 0]])
    calls = []

    def write(poses, env_ids):
        written[env_ids] = poses

    asset = SimpleNamespace(data=SimpleNamespace(default_root_state=defaults), write_root_pose_to_sim=write)

    def native_reset(env, env_ids, pose_range, velocity_range, asset_cfg):
        assert velocity_range == {} and asset_cfg.name == "cube"
        calls.append(env_ids.tolist())
        poses = defaults[env_ids, :7].clone()
        lower, upper = pose_range["x"]
        poses[:, 0] += torch.rand(len(env_ids)) * (upper - lower) + lower
        poses[:, :3] += origins[env_ids]
        write(poses, env_ids)

    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", SimpleNamespace(reset_root_state_uniform=native_reset))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(SceneEntityCfg=lambda name: SimpleNamespace(name=name)))
    rule = (RootReset(states=(Pose((0.1, 0, 0.831)), Pose((0.2, 0, 0.831), (sqrt(0.5), 0, 0, sqrt(0.5)))))
            if mode == "states" else RootReset(pose_range={"x": [-0.03, 0.03]}))
    world = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    world.index = {"world/cube": (Entity("cube", factory="box", reset=rule), Pose())}
    world.device, world.keys, world.rng = "cpu", {"world/cube": "cube"}, Random(42)
    world.scene = SimpleNamespace(env_origins=origins)
    world.asset = lambda path: asset
    rng_state = torch.random.get_rng_state().clone()
    ids = torch.arange(4)
    world._randomize_root_poses(ids)
    first = written.clone()
    world.rng.seed(42)
    world._randomize_root_poses(ids)
    torch.testing.assert_close(written, first)
    assert torch.equal(rng_state, torch.random.get_rng_state())
    local = written[:, :3] - origins
    if mode == "states":
        assert set(round(value, 3) for value in local[:, 0].tolist()) == {0.1, 0.2}
        for pose, position in zip(written, local):
            expected = rule.states[0] if position[0] < 0.15 else rule.states[1]
            torch.testing.assert_close(pose[3:], pose.new_tensor(expected.orientation))
        assert calls == []
    else:
        assert local[:, 0].min() >= -0.03 and local[:, 0].max() <= 0.03
        assert local[:, 0].unique().numel() == 4
        assert calls == [[0, 1, 2, 3], [0, 1, 2, 3]]
    world._randomize_root_poses(torch.tensor([1]))
    torch.testing.assert_close(written[[0, 2, 3]], first[[0, 2, 3]])
    unchanged = written.clone()
    world._randomize_root_poses(torch.tensor([], dtype=torch.long))
    torch.testing.assert_close(written, unchanged)


def _arm(name="arm", *, cameras=False):
    return load_assembly(ARM_CONFIG, cameras, overrides={"name": name})


@pytest.mark.parametrize("cameras", [False, True])
def test_generic_assemblies_expand_nested_files_with_isolated_overrides(tmp_path, cameras):
    from omegaconf import OmegaConf
    from projects.assembly_demo.layout import load_world_spec

    folder = tmp_path / "parts"
    folder.mkdir()
    template = {"schema_version": 1, "root": {
        "name": "prototype", "kind": "device", "factory": "box",
        "parameters": {"size": [0.2, 0.2, 0.2], "mass": 2},
        "initial_state": {"running": False},
        "children": [{"name": "tool", "factory": "box", "parameters": {"mass": 0.1}},
                     {"name": "camera", "kind": "sensor", "factory": "camera", "camera_only": True,
                      "mount": {"entity": ".."}}],
        "connections": [{"name": "tool_mount", "child": {"entity": "./tool"}, "parent": {"entity": "."}}],
    }}
    OmegaConf.save(OmegaConf.create(template), folder / "fixture.yaml")
    wrapper = {"schema_version": 1, "root": {
        "name": "station", "kind": "scene", "pose": {"position": [1, 0, 0]},
        "children": [{"name": "fixture", "assembly": "fixture.yaml",
                      "pose": {"position": [0, 1, 0]}, "parameters": {"mass": 4}}],
    }}
    OmegaConf.save(OmegaConf.create(wrapper), folder / "station.yaml")
    scene = {"schema_version": 1, "name": "lab", "entities": [
        {"name": "left", "assembly": "parts/station.yaml", "pose": {"position": [2, 0, 0]}},
        {"name": "right", "assembly": "parts/station.yaml", "pose": {"position": [-2, 0, 0]}},
    ]}
    destination = tmp_path / "scene.yaml"
    OmegaConf.save(OmegaConf.create(scene), destination)
    world = load_world_spec(destination, cameras)
    index = world.index()
    assert index["lab/left/fixture"][1].position == (2., 1., 0.)
    assert index["lab/right/fixture"][1].position == (-2., 1., 0.)
    left = index["lab/left/fixture"][0]
    right = index["lab/right/fixture"][0]
    assert left.kind == "device" and left.parameters == {"size": [0.2, 0.2, 0.2], "mass": 4}
    left.parameters["size"][0] = 9
    left.initial_state["running"] = True
    assert right.parameters["size"][0] == 0.2 and right.initial_state["running"] is False
    assert ("lab/left/fixture/camera" in index) is cameras
    assert {joint.child.entity for joint in world.ordered_connections()} == {
        "lab/left/fixture/tool", "lab/right/fixture/tool"}
    assert load_world_spec(destination, cameras).index()["lab/left/fixture"][0].parameters["size"][0] == 0.2
    assert load_assembly(folder / "fixture.yaml", overrides={"name": "standalone"}).name == "standalone"


def test_assembly_cycles_missing_files_and_invalid_overrides(tmp_path):
    from omegaconf import OmegaConf

    for name, target in (("first", "second"), ("second", "first")):
        data = {"schema_version": 1, "root": {"name": name, "assembly": f"{target}.yaml"}}
        OmegaConf.save(OmegaConf.create(data), tmp_path / f"{name}.yaml")
    with pytest.raises(ValueError, match="Cyclic assembly reference"):
        load_assembly(tmp_path / "first.yaml")
    with pytest.raises(FileNotFoundError):
        load_assembly(tmp_path / "missing.yaml")
    with pytest.raises(ValueError, match="Unknown assembly override fields"):
        load_assembly(ARM_CONFIG, overrides={"children": []})
    data = {"schema_version": 1, "root": {"name": "invalid", "assembly": ""}}
    OmegaConf.save(OmegaConf.create(data), tmp_path / "invalid.yaml")
    with pytest.raises(ValueError, match="nonempty file path"):
        load_assembly(tmp_path / "invalid.yaml")


@pytest.mark.parametrize("cameras", [False, True])
def test_tabletop_yaml_preserves_existing_description(cameras):
    from dataclasses import asdict
    import json

    reference = load_world_spec(scene_config_path("mobile"), cameras=cameras)
    index = reference.index()
    index["world/support"][0].parameters["dynamic"] = False
    index["world/arm"][0].parameters["fixed_base"] = True
    reference.connections = tuple(joint for joint in reference.connections if joint.name != "arm_mount")
    actual = load_world_spec(scene_config_path("tabletop"), cameras=cameras)
    assert json.loads(json.dumps(asdict(actual))) == json.loads(json.dumps(asdict(reference)))
    assert {path: node.kind for path, (node, _) in actual.index().items()} == {
        path: node.kind for path, (node, _) in reference.index().items()}
    assert actual.ordered_connections() == reference.ordered_connections()


def test_body_frames_are_declared_with_the_asset_and_reusable(tmp_path):
    from omegaconf import OmegaConf
    from shared.assembly.spec import BodyFrame

    data = {"schema_version": 1, "root": {"name": "arm", "kind": "robot", "factory": "franka",
            "frames": [{"name": "tcp", "body": "panda_hand", "pose": {"position": [0, 0, 0.1034]}}]}}
    path = tmp_path / "arm.yaml"
    OmegaConf.save(OmegaConf.create(data), path)
    left = load_assembly(path, overrides={"name": "left"})
    right = load_assembly(path, overrides={"name": "right"})
    index = WorldSpec(Scene("world", children=(left, right))).index()
    assert index["world/left"][0].frames == (BodyFrame("tcp", "panda_hand", Pose((0, 0, 0.1034))),)
    assert index["world/right"][0].frames == left.frames
    assert left is not right
    data["root"]["frames"][0]["unexpected"] = True
    OmegaConf.save(OmegaConf.create(data), path)
    with pytest.raises(ValueError, match="Unknown body frame fields"):
        load_assembly(path)


def test_body_frames_reject_duplicate_names_and_nonphysical_owners():
    from shared.assembly.spec import BodyFrame

    frame = BodyFrame("tcp", "panda_hand")
    with pytest.raises(ValueError, match="Duplicate body frame"):
        WorldSpec(Robot("arm", factory="franka", frames=(frame, frame))).index()
    for node in (Scene("group", frames=(frame,)), Sensor("camera", factory="camera", frames=(frame,))):
        with pytest.raises(ValueError, match="concrete non-sensor"):
            WorldSpec(node).index()
    with pytest.raises(ValueError, match="Invalid body frame name"):
        BodyFrame("bad/name")
    with pytest.raises(ValueError, match="relative body path"):
        BodyFrame("tcp", "../hand")


def test_isaac_registers_joint_parameters_and_live_frames_lazily(monkeypatch):
    import sys
    import torch
    from types import SimpleNamespace
    from shared.assembly import BodyFrame
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    class NativeArticulation:
        ready = False

        @property
        def body_names(self):
            if not self.ready:
                raise RuntimeError("Physics is not initialized")
            return ["hand"]

    class NativeRigidObject:
        pass

    class NativeCamera:
        pass

    monkeypatch.setitem(sys.modules, "isaaclab.assets", SimpleNamespace(
        Articulation=NativeArticulation, RigidObject=NativeRigidObject))
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", SimpleNamespace(Camera=NativeCamera))

    def combine(position, orientation, offset_position, offset_orientation):
        poses = [Pose(tuple(pos.tolist()), tuple(rot.tolist())).compose(
            Pose(tuple(local_pos.tolist()), tuple(local_rot.tolist())))
            for pos, rot, local_pos, local_rot in zip(position, orientation, offset_position, offset_orientation)]
        return position.new_tensor([pose.position for pose in poses]), orientation.new_tensor([pose.orientation for pose in poses])

    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", SimpleNamespace(combine_frame_transforms=combine))
    asset = NativeArticulation()
    half_turn = [0., 0., 0., 1.]
    # Native articulations publish ordered joint names alongside body names.
    asset.joint_names = ["shoulder", "elbow"]
    asset.data = SimpleNamespace(default_joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        soft_joint_pos_limits=torch.tensor([[[-1., 1.], [-2., 2.]]]).repeat(2, 1, 1),
        body_state_w=torch.tensor([[[10., 2., 3., 1., 0., 0., 0.]], [[20., 2., 3., *half_turn]]]))
    runtime = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    runtime.num_envs = 2
    runtime.index = {"world/arm": (Robot("arm", factory="franka", frames=(
        BodyFrame("tcp", "hand", Pose((0.1, 0, 0))), BodyFrame("tip", "hand", Pose((0, 0, 0.2))))), Pose())}
    runtime.keys = {"world/arm": "robot"}
    runtime.scene = {"robot": asset}
    runtime.signals = Signals()
    runtime._register_signals()
    defaults = runtime.signals.read("world/arm/joint_defaults")
    limits = runtime.signals.read("world/arm/joint_limits")
    assert defaults.shape == (2, 2) and limits.shape == (2, 2, 2)
    defaults.zero_()
    limits.zero_()
    assert asset.data.default_joint_pos.sum() > 0
    assert asset.data.soft_joint_pos_limits.abs().sum() > 0
    runtime.scene = SimpleNamespace(env_origins=torch.tensor([[10., 0, 0], [20., 0, 0]]))
    runtime.asset = lambda path: asset
    asset.ready = True
    pose = runtime.signals.read("world/arm/frames/tcp/pose")
    torch.testing.assert_close(pose[:, :3], torch.tensor([[0.1, 2., 3.], [-0.1, 2., 3.]]))
    torch.testing.assert_close(pose[:, 3:], asset.data.body_state_w[:, 0, 3:7])
    torch.testing.assert_close(runtime.signals.read("world/arm/frames/tip/position"),
                               torch.tensor([[0., 2., 3.2], [0., 2., 3.2]]))
    asset.data.body_state_w[:, 0, 2] += 1
    torch.testing.assert_close(runtime.signals.read("world/arm/frames/tcp/position")[:, 2], torch.tensor([4., 4.]))
    runtime.index["world/arm"][0].frames = (BodyFrame("missing", "absent"),)
    runtime.signals = Signals()
    runtime._register_signals()
    with pytest.raises(ValueError, match="explicit body"):
        runtime.signals.read("world/arm/frames/missing/pose")


def test_tabletop_yaml_overrides_and_invalid_fields(tmp_path):
    from pathlib import Path
    from omegaconf import OmegaConf
    from projects.assembly_demo.layout import load_world_spec

    source = Path(__file__).resolve().parents[1] / "projects/assembly_demo/tabletop.yaml"
    config = OmegaConf.load(source)
    config.entities[5].assembly = str(ARM_CONFIG)
    config.entities[6].pose.position = [0.2, 0.1, 0.9]
    destination = tmp_path / "scene.yaml"
    OmegaConf.save(config, destination)
    actual = load_world_spec(destination)
    assert actual.index()["world/cube_a"][1].position == (0.2, 0.1, 0.9)
    actual.index()["world/arm"][0].children[0].parameters["mass"] = 9
    assert load_world_spec(destination).index()["world/arm/tool"][0].parameters["mass"] == 0.05
    destination.write_text(source.read_text().replace("assemblies/arm.yaml", str(ARM_CONFIG))
                          .replace("[0.5, -0.65, 0.8]", "[0.6, -0.65, 0.8]"))
    relocated = load_world_spec(destination)
    assert relocated.index()["world/machine"][1].position == (0.6, -0.65, 0.8)
    assert relocated.ordered_connections()[0].parent_pose.position == (0.6, -0.65, 0.8)
    config.entities[6].positon = [1, 2, 3]
    OmegaConf.save(config, destination)
    with pytest.raises(ValueError, match="Unknown entity fields"):
        load_world_spec(destination)
    with pytest.raises(ValueError, match="Unknown entity fields"):
        load_world_spec(scene_config_path("mobile", destination))


@pytest.mark.parametrize("layout", ["tabletop", "mobile", "inspection"])
def test_all_layouts_accept_independent_custom_configs(layout, tmp_path):
    from omegaconf import OmegaConf
    from projects.assembly_demo.layout import scene_config_path

    source = scene_config_path(layout)
    assert source.name == f"{layout}.yaml"
    config = OmegaConf.load(source)
    for entity in config.entities:
        if "assembly" in entity:
            entity.assembly = str((source.parent / entity.assembly).resolve())
    floor = next(entity for entity in config.entities if entity.name == "floor")
    floor.pose.position = [0, 0, -0.08]
    destination = tmp_path / f"custom_{layout}.yaml"
    OmegaConf.save(config, destination)
    assert scene_config_path(layout, destination) == destination
    assert load_world_spec(scene_config_path(layout, destination)).index()["world/floor"][1].position == (0., 0., -0.08)
    assert load_world_spec(scene_config_path(layout)).index()["world/floor"][1].position == (0., 0., -0.05)
    with pytest.raises(ValueError, match="Unknown layout"):
        scene_config_path("missing")


def test_scene_config_is_validated_before_application_launch(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace
    from projects.assembly_demo.preview import main

    class NeverLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--enable_cameras", action="store_true")

        def __init__(self, args):
            pytest.fail("Invalid scene configuration must not launch Isaac")

    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=NeverLauncher))
    monkeypatch.setattr(sys, "argv", ["preview", "--layout", "mobile", "--scene-config", str(tmp_path / "scene.yaml")])
    with pytest.raises(FileNotFoundError):
        main()
    monkeypatch.setattr(sys, "argv", ["preview", "--scene-config", str(tmp_path / "missing.yaml")])
    with pytest.raises(FileNotFoundError):
        main()


def test_reusable_layouts_and_rotated_local_coordinates():
    q = (sqrt(.5), 0., 0., sqrt(.5))
    world = WorldSpec(Scene("world", pose=Pose((1., 0., 0.), q), children=(
        Entity("fixture", pose=Pose((1., 0., 0.))),)))
    assert world.index()["world/fixture"][1].position == pytest.approx((1., 1., 0.))
    for name in ("tabletop", "mobile", "inspection"):
        layout = load_world_spec(scene_config_path(name), cameras=True)
        assert layout.index()
        layout.ordered_connections()
    assert [j.name for j in load_world_spec(scene_config_path("mobile")).ordered_connections()] == ["machine_anchor", "arm_mount", "world__arm__tool_mount"]
    nested = WorldSpec(Scene("lab", children=(Scene("station", children=(_arm("left", cameras=True), _arm("right", cameras=True))),)))
    assert nested.index()
    assert nested.resolve_mount("lab/station/left/wrist_camera", Mount("..", "panda_hand")) == Mount("lab/station/left", "panda_hand")
    joints = nested.ordered_connections()
    assert {joint.child.entity for joint in joints} == {"lab/station/left/tool", "lab/station/right/tool"}
    assert {joint.parent.entity for joint in joints} == {"lab/station/left", "lab/station/right"}
    assert len({joint.name for joint in joints}) == 2
    assert all(joint.parent.body == "panda_hand" and joint.parent_pose.position == (0.08, 0., 0.03) for joint in joints)
    left, right = nested.root.children[0].children
    left.parameters["fixed_base"] = False
    left.children[0].parameters["mass"] = 1.0
    assert right.parameters["fixed_base"] is True
    assert right.children[0].parameters["mass"] == 0.05
    relocated = WorldSpec(Scene("other", children=(_arm("renamed", cameras=True),)))
    assert relocated.ordered_connections()[0].child == Mount("other/renamed/tool")


def test_internal_connections_cannot_escape_or_conflict_with_external_joints():
    for parent in (None, Mount("world/outside"), Mount("..")):
        entity = Robot("arm", factory="test:factory", children=(Entity("tool", factory="test:factory"),),
                       connections=(FixedJoint("tool_mount", Mount("./tool"), parent),))
        with pytest.raises(ValueError, match="parent|relative|escapes"):
            WorldSpec(Scene("world", children=(entity,))).index()
    with pytest.raises(ValueError, match="Multiple attachment parents"):
        WorldSpec(Scene("world", children=(_arm(),)), (
            FixedJoint("duplicate", Mount("world/arm/tool"), Mount("world/arm", "panda_hand")),)).index()
    broken = _arm()
    broken.connections = (FixedJoint("missing", Mount("./missing"), Mount(".")),)
    with pytest.raises(ValueError, match="concrete asset"):
        WorldSpec(Scene("world", children=(broken,))).index()
    broken.connections = (FixedJoint("", Mount("./tool"), Mount(".")),)
    with pytest.raises(ValueError, match="internal joint name"):
        WorldSpec(Scene("world", children=(broken,))).index()


def test_invalid_graphs_fail_before_simulator_start():
    node = Entity("a")
    with pytest.raises(ValueError, match="reused|cyclic"):
        WorldSpec(Scene("world", children=(node, node))).index()
    with pytest.raises(ValueError, match="Duplicate"):
        WorldSpec(Scene("world", children=(Entity("a"), Entity("a")))).index()
    a, b = Entity("a", factory="test:factory"), Robot("b", factory="test:factory")
    with pytest.raises(ValueError, match="Cyclic"):
        WorldSpec(Scene("world", children=(a, b)), (
            FixedJoint("ab", Mount("world/a"), Mount("world/b")),
            FixedJoint("ba", Mount("world/b"), Mount("world/a")),)).index()
    with pytest.raises(ValueError, match="concrete asset"):
        WorldSpec(Scene("world", children=(Sensor("cam", factory="test:factory", mount=Mount("missing")),))).index()
    with pytest.raises(ValueError, match="normalized"):
        Pose(orientation=(2., 0., 0., 0.))


def test_observation_is_lazy_and_aliased_reads_are_cached():
    calls = []
    signals = Signals()
    signals.register("robot/joints", Signal(lambda: calls.append("joints") or [1, 2]))
    signals.register("camera/rgb", Signal(lambda: pytest.fail("unselected image must not be read")))
    signals.describe()
    assert calls == []
    obs = Observation({"a": "robot/joints", "b": "robot/joints"})
    assert obs.read(signals) == {"a": [1, 2], "b": [1, 2]}
    assert calls == ["joints"]
    with pytest.raises(KeyError):
        Observation({"a": "robot/joints", "typo": "missing"}).read(signals)
    assert calls == ["joints"], "Validate all fields before reading any"


def test_sensor_preparation_is_batched_validated_and_optional():
    events = []
    signals = Signals(prepare=lambda paths: events.append(("prepare", paths)))
    for path in ("left/rgb", "right/rgb", "switch/state/pressed"):
        signals.register(path, Signal(lambda source=path: events.append(("read", source)) or source))
    fields = {"left": "left/rgb", "alias": "left/rgb", "right": "right/rgb"}
    assert Observation(fields).read(signals) == {alias: path for alias, path in fields.items()}
    assert events == [("prepare", ("left/rgb", "right/rgb")), ("read", "left/rgb"), ("read", "right/rgb")]
    events.clear()
    with pytest.raises(KeyError):
        signals.read_many(("left/rgb", "missing"))
    assert events == []
    assert signals.read("switch/state/pressed") == "switch/state/pressed"
    assert events == [("prepare", ("switch/state/pressed",)), ("read", "switch/state/pressed")]


@pytest.mark.parametrize("layout", ["tabletop", "mobile", "inspection"])
def test_scene_sensor_roles_and_configured_object_observations(layout):
    from projects.assembly_demo.components import load_observation_spec
    from projects.assembly_demo.layout import scene_config_path

    world = load_world_spec(scene_config_path(layout), cameras=True)
    sensors = [node for node, _ in world.index().values() if isinstance(node, Sensor)]
    assert sensors and all(type(node) is Sensor and node.kind == "sensor" for node in sensors)
    state = load_observation_spec(scene_config_path(layout))
    assert state.fields["switch_pressed"] == "world/switch/state/pressed"
    assert state.fields["machine_running"] == "world/machine/state/running"
    assert state.fields["indicator_on"] == "world/machine/indicator/state/on"
    assert all(not path.endswith("/rgb") for path in state.fields.values())
    vision = load_observation_spec(scene_config_path(layout), "vision")
    assert vision.fields == {**state.fields, "overview_rgb": "world/overview/rgb"}
    with pytest.raises(ValueError, match="Unknown observation profile"):
        load_observation_spec(scene_config_path(layout), "missing")


def test_sensor_role_does_not_require_a_camera_factory(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from projects.assembly_demo.assets import FACTORIES
    from projects.assembly_demo.layout import load_world_spec

    monkeypatch.setitem(FACTORIES, "measurement", "test:sensor_factory")
    config = {"schema_version": 1, "name": "world", "entities": [
        {"name": "body", "factory": "box"},
        {"name": "measurement", "kind": "sensor", "factory": "measurement", "mount": {"entity": "world/body"}}]}
    path = tmp_path / "sensor.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    sensor = load_world_spec(path).index()["world/measurement"][0]
    assert type(sensor) is Sensor and sensor.factory == "measurement"
    with pytest.raises(ValueError, match="Sensor needs"):
        WorldSpec(Scene("world", children=(Sensor("missing"),))).index()
    with pytest.raises(ValueError, match="non-sensor"):
        WorldSpec(Scene("world", children=(Sensor("first", factory="test:sensor_factory"),
            Sensor("second", factory="test:sensor_factory", mount=Mount("world/first"))))).index()


def test_configured_scene_switch_observations_and_partial_reset():
    from copy import deepcopy
    from projects.assembly_demo.components import SwitchMachine, load_observation_spec
    from projects.assembly_demo.layout import scene_config_path

    class SceneState:
        step_dt = 0.01
        num_envs = 2
        def environment_ids(self, ids): return list(range(2)) if ids is None else list(ids)
        def __init__(self):
            self.index = load_world_spec(scene_config_path()).index()
            self.state = {path: [deepcopy(node.initial_state) for _ in range(2)]
                          for path, (node, _) in self.index.items()}
            self.signals = Signals()
            for path, values in self.state.items():
                for key in values[0]:
                    self.signals.register(path + "/state/" + key, Signal(
                        lambda source=path, field=key: [item[field] for item in self.state[source]], "semantic"))
        def set_state(self, path, key, value, env_ids=None):
            for index in range(2) if env_ids is None else env_ids:
                self.state[path][index][key] = value
        def reset(self, env_ids):
            for path, (node, _) in self.index.items():
                for index in range(2) if env_ids is None else env_ids:
                    self.state[path][index] = deepcopy(node.initial_state)
        def advance(self):
            pass
    selection = load_observation_spec(scene_config_path())
    selection.fields.pop("joints")
    world = SceneState()
    env = AssemblyEnv(world, selection, systems=(SwitchMachine(),))
    assert {alias: value.tolist() for alias, value in env.reset()[0].items()} == {alias: [False, False] for alias in selection.fields}
    world.set_state("world/switch", "pressed", True, [0])
    assert {alias: value.tolist() for alias, value in env.step()[0].items()} == {alias: [True, False] for alias in selection.fields}
    world.set_state("world/switch", "pressed", True, [1])
    assert {alias: value.tolist() for alias, value in env.step()[0].items()} == {alias: [True, True] for alias in selection.fields}
    assert {alias: value.tolist() for alias, value in env.reset([0])[0].items()} == {alias: [False, True] for alias in selection.fields}


@pytest.mark.parametrize("fields", [[], {"": "source"}, {"alias": ""}, {"alias": 1}])
def test_invalid_observation_specs(fields):
    with pytest.raises(ValueError, match="nonempty"):
        Observation(fields)


def test_camera_render_products_start_disabled(monkeypatch):
    import sys
    from types import SimpleNamespace
    from shared.backends.isaac.preview_world import _on_demand_camera

    toggles = []
    bindings = []
    product = SimpleNamespace(path="/render/product", hydra_texture=SimpleNamespace(
        set_updates_enabled=lambda enabled: toggles.append(enabled)))
    annotator = SimpleNamespace(detach=lambda paths: bindings.append(("detach", paths)),
                                attach=lambda source: bindings.append(("attach", source.path)),
                                get_data=lambda: SimpleNamespace(shape=(120, 160, 4)))
    def render_product(path, *, resolution, force_new):
        assert path == "/camera" and resolution == (160, 120) and force_new is False
        return product
    monkeypatch.setitem(sys.modules, "omni.replicator.core", SimpleNamespace(
        create=SimpleNamespace(render_product=render_product)))
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(replicator=SimpleNamespace(
        core=sys.modules["omni.replicator.core"])))
    monkeypatch.setitem(sys.modules, "omni.replicator", sys.modules["omni"].replicator)
    class NativeCamera:
        cfg = SimpleNamespace(width=160, height=120)
        render_product_paths = [product.path]
        _sensor_prims = [SimpleNamespace(GetPath=lambda: "/camera")]
        _rep_registry = {"rgb": [annotator]}
        def _initialize_impl(self):
            pass
    camera = _on_demand_camera(NativeCamera)()
    camera._initialize_impl()
    assert toggles == [False]
    camera.set_rendering(True)
    camera.set_rendering(False)
    assert toggles == [False, True, False]
    camera.set_rendering(True)
    camera.set_rendering(False)
    assert bindings == [("detach", [product.path]), ("attach", product.path)] * 2
    assert camera.has_rendered_data()
    annotator.get_data = lambda: SimpleNamespace(shape=(0,))
    assert not camera.has_rendered_data()
    annotator.get_data = lambda: {"data": SimpleNamespace(shape=(120, 160, 4))}
    assert camera.has_rendered_data()


def test_only_requested_cameras_render_and_images_are_cached():
    from types import SimpleNamespace
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    events = []
    class CameraStub:
        def __init__(self, path):
            self.path = path
        def has_rendered_data(self):
            return True
        def set_rendering(self, enabled):
            events.append((self.path, enabled))
        def update(self, dt, force_recompute):
            assert dt == 0.0 and force_recompute
            events.append((self.path, "capture"))
    world = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    world._sensor_epoch, world._camera_samples = 0, {}
    world._image_sources = {"left/rgb": "left", "right/rgb": "right"}
    cameras = {path: CameraStub(path) for path in ("left", "right")}
    world.asset = cameras.__getitem__
    world.sim = SimpleNamespace(render=lambda: events.append("render"))
    world.signals = Signals(prepare=world._prepare_signals)
    for path in (*world._image_sources, "switch/state/pressed", "left/position"):
        world.signals.register(path, Signal(lambda source=path: source))
    world.signals.describe()
    Observation({"switch": "switch/state/pressed", "pose": "left/position"}).read(world.signals)
    assert events == []
    world.signals.read("left/rgb")
    assert events == [("left", True), "render", "render", "render", ("left", "capture"), ("left", False)]
    events.clear()
    world.signals.read("left/rgb")
    assert events == []
    world._sensor_epoch += 1
    Observation({"left": "left/rgb", "right": "right/rgb"}).read(world.signals)
    assert events == [("left", True), ("right", True), "render", "render", "render",
                      ("left", "capture"), ("right", "capture"), ("left", False), ("right", False)]
    events.clear()
    world._sensor_epoch += 1
    world.signals.read("right/rgb")
    assert events == [("right", True), "render", ("right", "capture"), ("right", False)]
    events.clear()
    def fail_render():
        raise RuntimeError("render failed")
    world.sim.render = fail_render
    world._sensor_epoch += 1
    with pytest.raises(RuntimeError, match="render failed"):
        world.signals.read("right/rgb")
    assert events == [("right", True), ("right", False)]
    assert world._camera_samples["right"] != world._sensor_epoch
    events.clear()
    world.sim.render = lambda: events.append("render")
    readiness = iter((False, False, True))
    cameras["right"].has_rendered_data = lambda: next(readiness)
    world.signals.read("right/rgb")
    assert events == [("right", True), "render", "render", "render", ("right", "capture"), ("right", False)]
    events.clear()
    world._sensor_epoch += 1
    cameras["right"].has_rendered_data = lambda: False
    with pytest.raises(RuntimeError, match="did not become ready"):
        world.signals.read("right/rgb")
    assert events == [("right", True)] + ["render"] * 21 + [("right", False)]
    assert world._camera_samples["right"] != world._sensor_epoch


@pytest.mark.parametrize("gui", [False, True])
def test_physics_advance_renders_only_the_gui_without_sensor_demand(gui):
    from types import SimpleNamespace
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    events = []
    world = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    world.decimation, world.elapsed_steps, world._sensor_epoch = 2, 0, 0
    world.scene = SimpleNamespace(write_data_to_sim=lambda: events.append("write"),
                                 update=lambda dt: events.append(("update", dt)))
    world.sim = SimpleNamespace(step=lambda render: events.append(("physics", render)),
                                get_physics_dt=lambda: 0.005, has_gui=lambda: gui,
                                has_rtx_sensors=lambda: True, render=lambda: events.append("render"))
    world._sync_cameras = lambda: events.append("mounts")
    world.advance()
    assert events == ["write", ("physics", False), ("update", 0.005)] * 2 + ["mounts"] + (["render"] if gui else [])
    assert world.elapsed_steps == 1 and world._sensor_epoch == 1


def test_standalone_preview_component_order_and_optional_task():
    events = []
    class FakeWorld:
        step_dt = .02
        num_envs = 2
        signals = Signals()
        def environment_ids(self, ids): return list(range(2)) if ids is None else list(ids)
        def reset(self, ids): events.append(("world_reset", ids))
        def advance(self): events.append("physics")
    class Component:
        def reset(self, w, ids): events.append(("component_reset", ids))
        def apply(self, w, a): events.append(("action", a))
        def step(self, w, dt): events.append(("system", dt))
        def evaluate(self, w):
            events.append("task")
            return Evaluation([1., 1.], [False, False], [False, False], {"success": [False, False]})
    world = FakeWorld()
    world.signals.register("state", Signal(lambda: events.append("observation") or [7, 7]))
    component = Component()
    env = AssemblyEnv(world, Observation({"x": "state"}), action=component, task=component, systems=(component,))
    with pytest.raises(RuntimeError): env.step(1)
    with pytest.raises(RuntimeError): env.reset([0])
    observation, info = env.reset()
    assert observation["x"].tolist() == [7, 7] and info == {}
    events.clear()
    result = env.step(3)
    assert events == [("action", 3), "physics", ("system", .02), "task", "observation"]
    assert result[1].tolist() == [1., 1.]
    events.clear()
    env.reset([1])
    assert events == [("world_reset", [1])] + [("component_reset", [1])] * 3 + ["observation"]
    env = AssemblyEnv(world, Observation({}))
    env.reset()
    assert env.step()[1].tolist() == [0., 0.]
    with pytest.raises(ValueError): env.step(3)


def _make_training_environment(parameters=None, num_envs=2, seed=0, horizon=2):
    """CPU simulator protocol probe used by actual RLinf subprocess transport."""
    from random import Random
    from types import SimpleNamespace
    import torch
    from shared.backends.isaac.environment import Environment

    class Native:
        def __init__(self):
            self.num_envs, self.device = num_envs, "cpu"
            self.action_manager = SimpleNamespace(total_action_dim=1)
            self.values = torch.zeros(num_envs)
            self.reset_calls = []
            self.rng = Random(seed)
            self.closed = 0

        def reset(self, env_ids=None, seed=None):
            if seed is not None:
                self.rng.seed(seed)
            ids = list(range(num_envs)) if env_ids is None else torch.as_tensor(env_ids).tolist()
            self.reset_calls.append(ids)
            self.values[ids] = 0
            return {"states": self.values[:, None]}, {}

        def step(self, action):
            if parameters is not None and parameters.get("fail_on_step", False):
                raise RuntimeError("physics contract probe failed")
            self.values += 1 + action[:, 0]
            success, failure = self.values >= 2, self.values < 0
            return {"states": self.values[:, None]}, self.values, success | failure, torch.zeros_like(success), {
                "metrics": {"success": success, "failure": failure}}

        def close(self):
            self.closed += 1

    class Owner:
        def __init__(self):
            self.closed = 0
        def close(self):
            self.closed += 1

    environment = Environment(Native())
    return environment, Owner() if parameters is None or parameters.get("resource_owner", True) else None


def _assembly_training_config():
    from omegaconf import OmegaConf

    return OmegaConf.create({"seed": 3, "auto_reset": True, "ignore_terminations": False,
                            "use_rel_reward": False, "max_episode_steps": 2,
                            "video_cfg": {"save_video": False}, "init_params": {
                                "id": "Workspace-Assembly-Contract-Test-v0",
                                "task_description": "CPU interface validation, not a robot task",
                                "obs_dim": 1, "action_dim": 1}})


@pytest.fixture
def rlinf_source_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "third_party/RLinf"))


@pytest.mark.parametrize("resource_owner", [False, True])
def test_environment_factory_preserves_optional_resource_owner(resource_owner):
    from shared.integrations.rlinf.tensors import _make_environment

    cfg = _assembly_training_config()
    cfg.init_params.resource_owner = resource_owner
    environment, owner = _make_environment(cfg.init_params, 2, 8, 2, factory=_make_training_environment)
    assert (owner is not None) == resource_owner
    import random
    assert environment.num_envs == 2
    assert environment.native.rng.random() == random.Random(8).random()
    environment.close()
    if owner is not None:
        owner.close()


@pytest.mark.parametrize("numpy_actions", [False, True])
def test_isaac_thin_entrypoint_uses_rlinf_native_subprocess(rlinf_source_path, monkeypatch, numpy_actions):
    import torch
    from rlinf.envs import get_env_cls
    from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS
    from shared.integrations.rlinf.isaac import IsaacAssemblyRLinfEnv, _ENVIRONMENT_FACTORIES, register_isaac_assembly_env

    cfg = _assembly_training_config()
    monkeypatch.setitem(REGISTER_ISAACLAB_ENVS, cfg.init_params.id, None)
    monkeypatch.setitem(_ENVIRONMENT_FACTORIES, cfg.init_params.id, None)
    register_isaac_assembly_env(cfg.init_params.id, factory=_make_training_environment)
    register_isaac_assembly_env(cfg.init_params.id, factory=_make_training_environment)
    with pytest.raises(TypeError, match="callable"):
        register_isaac_assembly_env(cfg.init_params.id, factory="not a function")
    with pytest.raises(ValueError, match="already has"):
        register_isaac_assembly_env(cfg.init_params.id, factory=lambda *args: None)
    assert get_env_cls("isaaclab", cfg) is IsaacAssemblyRLinfEnv
    environment = IsaacAssemblyRLinfEnv(cfg, 2, 5, 1, None)
    try:
        assert torch.device(environment.device) == torch.device("cpu") and environment.seed == 8
        environment.reset(seed=17)
        # Chunks are supported; the environment count and action width are not.
        for invalid in (torch.zeros(2, 1, 2), torch.zeros(1, 1, 1),
                torch.full((2, 1, 1), float("nan"))):
            with pytest.raises(ValueError, match="finite actions"): environment.chunk_step(invalid)
        assert not environment.elapsed_steps.any()
        actions = torch.tensor([[[1.]], [[0.]]])
        observations, rewards, terminated, truncated, infos = environment.chunk_step(
            actions.numpy() if numpy_actions else actions)
        assert observations[-1]["states"].flatten().tolist() == [0., 1.]
        assert rewards.tolist() == [[2.], [1.]]
        assert terminated.tolist() == [[True], [False]] and not truncated.any()
        assert infos[-1]["final_observation"]["states"].flatten().tolist() == [2., 1.]
        terminal_info = infos[-1]["final_info"]
        assert terminal_info["metrics"]["success"].tolist() == [True, False]
        assert terminal_info["episode"]["success_once"].tolist() == [True, False]
        assert terminal_info["episode"]["episode_len"].tolist() == [1, 1]
        assert environment.elapsed_steps.tolist() == [0, 1]
        assert environment.returns.tolist() == [0., 1.]
        observations, _, _, truncated, infos = environment.chunk_step(torch.zeros(2, 1, 1))
        assert truncated.tolist() == [[False], [True]]
        assert observations[-1]["states"].flatten().tolist() == [1., 0.]
        assert infos[-1]["final_info"]["episode"]["return"].tolist() == [1., 3.]
        assert terminal_info["episode"]["return"].tolist() == [2., 1.]
    finally:
        environment.close()
        environment.close()
    assert not environment.env.isaac_lab_process.is_alive()


@pytest.mark.parametrize("ending", ["timeout", "failure"])
def test_native_episode_uses_task_success_and_failure_not_positive_rewards(rlinf_source_path, monkeypatch, ending):
    import torch
    from shared.integrations.rlinf.isaac import IsaacAssemblyRLinfEnv, _ENVIRONMENT_FACTORIES

    cfg = _assembly_training_config()
    if ending == "timeout":
        cfg.max_episode_steps = 1
    monkeypatch.setitem(_ENVIRONMENT_FACTORIES, cfg.init_params.id, _make_training_environment)
    environment = IsaacAssemblyRLinfEnv(cfg, 2, 0, 1, None)
    try:
        actions = torch.zeros(2, 1)
        if ending == "failure":
            actions[0, 0] = -2
        _, _, terminated, truncated, info = environment.step(actions)
        final_info = info["final_info"]
        assert not final_info["episode"]["success_once"].any()
        assert final_info["metrics"]["success"].tolist() == [False, False]
        if ending == "timeout":
            assert not terminated.any() and truncated.all()
            assert not final_info["episode"]["fail_once"].any()
        else:
            assert terminated.tolist() == [True, False] and not truncated.any()
            assert final_info["episode"]["fail_once"].tolist() == [True, False]
    finally:
        environment.close()


def test_multi_input_network_does_not_require_flat_states(rlinf_source_path):
    import torch
    from shared.integrations.rlinf.tensors import _wrap_observations

    class ListInputNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output = torch.nn.Linear(10, 8)

        def forward(self, inputs):
            joints, image = inputs
            return self.output(torch.cat((joints, image.mean(dim=(-1, -2))), dim=-1))

    raw = {"joints": torch.ones(2, 7), "image": torch.ones(2, 3, 8, 8)}
    observation = _wrap_observations(raw, 2)
    assert set(observation) == {"joints", "image"}
    network = ListInputNetwork()
    actions = network([observation["joints"], observation["image"]])
    assert actions.shape == (2, 8)
    actions.sum().backward()
    assert network.output.weight.grad.abs().sum() > 0
    raw["joints"].zero_()
    assert observation["joints"].sum() == 14
    with pytest.raises(ValueError, match="leading dimension"):
        _wrap_observations({"image": torch.zeros(1, 3, 8, 8)}, 2)
    with pytest.raises(ValueError, match="finite"):
        _wrap_observations({"joints": torch.full((2, 7), float("nan"))}, 2)


def test_rlinf_action_conversion_does_not_rescale():
    import torch
    from shared.integrations.rlinf.tensors import _prepare_actions

    raw = torch.tensor([[[0.25]], [[-0.5]]])
    converted = _prepare_actions(raw.numpy(), (2, 1, 1), "cpu")
    torch.testing.assert_close(converted, raw)


@pytest.mark.parametrize("field,value", [
    ("auto_reset", False), ("ignore_terminations", True), ("use_rel_reward", True),
    ("video_cfg.save_video", True), ("max_episode_steps", 0),
    ("init_params.action_dim", 0),
    ("process_isolation", "true"),
    ("process_timeout_s", 300),
])
def test_rlinf_invalid_config_is_rejected_before_subprocess(monkeypatch, field, value, rlinf_source_path):
    from omegaconf import OmegaConf
    from rlinf.envs.sim.isaaclab.isaaclab_env import IsaaclabBaseEnv
    from shared.integrations.rlinf.isaac import IsaacAssemblyRLinfEnv, _ENVIRONMENT_FACTORIES

    cfg = _assembly_training_config()
    OmegaConf.update(cfg, field, value)
    monkeypatch.setitem(_ENVIRONMENT_FACTORIES, cfg.init_params.id, _make_training_environment)
    def fail_initialization(*args, **kwargs):
        pytest.fail("Invalid configuration must not create a simulator subprocess")
    monkeypatch.setattr(IsaaclabBaseEnv, "__init__", fail_initialization)
    with pytest.raises(ValueError):
        IsaacAssemblyRLinfEnv(cfg, 2, 0, 1, None)


def test_joint_action_uses_runtime_control_not_native_assets():
    from projects.assembly_demo.components import JointTargetAction

    calls = []
    class Runtime:
        def set_joint_position_target(self, path, target, env_ids=None):
            calls.append((path, target, env_ids))
    runtime = Runtime()
    action = JointTargetAction("world/robot")
    action.apply(runtime, [[0.1, 0.2]])
    action.apply(runtime, None)
    assert calls == [("world/robot", [[0.1, 0.2]], None)]


def test_shared_runtime_runs_project_logic_without_native_objects():
    from shared.backends.isaac.preview_runtime import RuntimeWorld
    from projects.assembly_demo.components import JointTargetAction, SwitchMachine, load_observation_spec
    from projects.assembly_demo.layout import scene_config_path

    class MemoryRuntime(RuntimeWorld):
        def __init__(self, description):
            super().__init__(description, num_envs=2, step_dt=0.01)
            self.targets = [[0.0], [0.0]]
            self.register_signal("world/arm/joint_position", Signal(lambda: [row[:] for row in self.targets]))
        def set_joint_position_target(self, path, target, env_ids=None):
            assert path == "world/arm"
            for env_id, row in zip(self.environment_ids(env_ids), target):
                self.targets[env_id] = list(row)
        def advance(self):
            pass
        def reset(self, env_ids=None):
            for env_id in self.environment_ids(env_ids):
                self.targets[env_id] = [0.0]
            self.reset_semantic_state(env_ids)
    description = load_world_spec(scene_config_path())
    assert description.index()["world/arm"][0].factory == "franka"
    left, right = MemoryRuntime(description), MemoryRuntime(description)
    selection = load_observation_spec(scene_config_path())
    env = AssemblyEnv(left, selection, action=JointTargetAction("world/arm"), systems=(SwitchMachine(),))
    assert env.reset()[0]["joints"].tolist() == [[0.0], [0.0]]
    left.set_state("world/switch", "pressed", True, [1])
    result = env.step([[0.1], [0.2]])
    assert result[0]["joints"].flatten().tolist() == pytest.approx([0.1, 0.2])
    assert result[0]["machine_running"].tolist() == [False, True]
    assert env.reset([0])[0]["joints"].flatten().tolist() == pytest.approx([0.0, 0.2])
    assert left.signals.read("world/machine/state/running") == [False, True]
    assert right.signals.read("world/machine/state/running") == [False, False]
    state = left.signals.read("world/machine/semantic_state")
    state[1]["running"] = False
    assert left.signals.read("world/machine/state/running") == [False, True]
    left.set_state("world/machine", "running", {"values": []}, [0, 1])
    left.semantic_state["world/machine"][0]["running"]["values"].append(1)
    assert left.semantic_state["world/machine"][1]["running"] == {"values": []}
    for ids in ([True], [0.0], [0, 0], [-1], [2], [[0]], 0):
        with pytest.raises(ValueError):
            left.set_state("world/switch", "pressed", True, ids)
    with pytest.raises(KeyError, match="Undeclared"):
        left.set_state("world/switch", "missing", True)
    assert left.environment_ids([]) == []
    with pytest.raises(TypeError):
        RuntimeWorld(description, num_envs=2, step_dt=0.01)


def test_isaac_joint_control_adapts_and_validates_runtime_targets():
    import torch
    from types import SimpleNamespace
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    calls = []
    asset = SimpleNamespace(num_joints=2, data=SimpleNamespace(joint_pos=torch.zeros(2, 2)),
        set_joint_position_target=lambda target, env_ids: calls.append((target.clone(), env_ids.clone())))
    runtime = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    runtime.keys = {"world/arm": "arm", "world/object": "object"}
    runtime.scene = SimpleNamespace(articulations={"arm": asset})
    runtime.asset = lambda path: asset
    runtime.device, runtime.num_envs = "cpu", 2
    runtime.set_joint_position_target("world/arm", [[0.1, 0.2], [0.3, 0.4]])
    assert calls[0][0].shape == (2, 2) and calls[0][0].dtype == torch.float32
    assert calls[0][1].tolist() == [0, 1]
    runtime.set_joint_position_target("world/arm", [[0.5, 0.6]], [1])
    assert calls[1][1].tolist() == [1]
    for target in ([0.1, 0.2], [[0.1]], [[float("nan"), 0.2], [0.3, 0.4]]):
        with pytest.raises(ValueError, match="finite with shape"):
            runtime.set_joint_position_target("world/arm", target)
    with pytest.raises(TypeError, match="articulation"):
        runtime.set_joint_position_target("world/object", [[0.1]])
    with pytest.raises(ValueError):
        runtime.set_joint_position_target("world/arm", [[0.1, 0.2]], [2])
    assert len(calls) == 2


def test_backend_factory_resolution_requires_an_explicit_registry(monkeypatch):
    import sys
    from types import SimpleNamespace
    from shared.backends.isaac.preview_world import IsaacAssemblyWorld

    runtime = IsaacAssemblyWorld.__new__(IsaacAssemblyWorld)
    node = Entity("box", factory="box", parameters={"size": [0.1, 0.2, 0.3]})
    original = {"configured": True}
    def factory(size):
        size[0] = 9
        return original
    monkeypatch.setitem(sys.modules, "test_factory_module", SimpleNamespace(create=factory))
    runtime.factories = {"box": "test_factory_module:create"}
    result = runtime._asset_config(node)
    result["configured"] = False
    assert original == {"configured": True}
    assert node.parameters == {"size": [0.1, 0.2, 0.3]}
    for asset in ("missing", "test_factory_module:create"):
        with pytest.raises(ValueError, match="No Isaac factory registered"):
            runtime._asset_config(Entity("object", factory=asset))


def test_new_framework_does_not_export_legacy_aliases(tmp_path):
    import shared.assembly as assembly
    from omegaconf import OmegaConf
    from projects.assembly_demo import components
    from projects.assembly_demo.layout import load_world_spec

    assert not hasattr(components, "load_observation")
    assert not hasattr(assembly, "Camera")
    assert not hasattr(assembly, "World")
    assert not hasattr(assembly, "StepResult")
    for internal in ("AssemblyEnv", "RuntimeWorld", "Task", "ActionAdapter", "Evaluation", "System"):
        assert not hasattr(assembly, internal)
    from projects.assembly_demo import layout
    assert not hasattr(layout, "build")
    assert not hasattr(layout, "load_scene")
    config = {"schema_version": 1, "name": "world", "entities": [
        {"name": "view", "kind": "camera", "factory": "camera"}]}
    path = tmp_path / "old_role.yaml"
    OmegaConf.save(OmegaConf.create(config), path)
    with pytest.raises(ValueError, match="Unknown entity kind"):
        load_world_spec(path, cameras=True)
