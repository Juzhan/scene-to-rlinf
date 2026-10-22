"""Author reusable robot instances by name without copying their whole definition."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
import pytest

from shared.assembly.layout import load_assembly, load_world_spec


ROOT = Path(__file__).resolve().parents[1]
ARM = ROOT / "projects/assembly_demo/assemblies/arm.yaml"
FACTORIES = {name: lambda **_: None for name in ("franka", "camera", "box")}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(value), path)
    return path


def test_two_instances_customize_camera_tool_and_connection_without_leaking(tmp_path):
    source = ARM.read_text()
    changes = {"wrist_camera": {"parameters": {"width": 320, "height": 240},
                                 "pose": {"position": [0.1, 0.0, 0.05]}},
               "tool": {"parameters": {"mass": 0.2}, "tags": ["inspection_tool"]}}
    original = deepcopy(changes)
    path = save(tmp_path / "scene.yaml", {"schema_version": 1, "name": "room", "entities": [
        {"name": "left", "assembly": str(ARM), "overrides": changes,
         "connection_overrides": {"tool_mount": {"parent_pose": {"position": [0.12, 0.0, 0.04]}}}},
        {"name": "right", "assembly": str(ARM)},
    ]})
    world = load_world_spec(path, cameras=True, factories=FACTORIES)
    index = world.index()
    left, right = index["room/left/wrist_camera"][0], index["room/right/wrist_camera"][0]
    assert left.parameters == {"width": 320, "height": 240}
    assert right.parameters == {}
    assert left.pose.position == (0.1, 0.0, 0.05) and right.pose.position == (0.2, 0.0, 0.06)
    assert left.pose.orientation == right.pose.orientation == (0., 1., 0., 0.)
    assert world.resolve_mount("room/left/wrist_camera", left.mount).entity == "room/left"
    assert world.resolve_mount("room/right/wrist_camera", right.mount).entity == "room/right"
    assert index["room/left/tool"][0].parameters == {"size": [0.03, 0.03, 0.04], "mass": .2}
    assert index["room/right/tool"][0].parameters["mass"] == .05
    joints = {joint.child.entity: joint for joint in world.ordered_connections()}
    assert joints["room/left/tool"].parent_pose.position == (.12, 0., .04)
    assert joints["room/right/tool"].parent_pose.position == (.08, 0., .03)
    index["room/left/tool"][0].parameters["size"][0] = 99
    assert index["room/right/tool"][0].parameters["size"][0] == .03
    assert ARM.read_text() == source and changes == original


def test_nested_source_paths_and_outer_overrides_win_before_camera_filtering(tmp_path):
    inner = save(tmp_path / "parts/inner/robot.yaml", {
        "schema_version": 1, "root": {"name": "arm", "factory": "franka", "children": [
            {"name": "eye", "kind": "sensor", "factory": "camera", "camera_only": True,
             "mount": {"entity": "..", "body": "panda_hand"},
             "parameters": {"width": 160, "height": 120}},
        ]},
    })
    outer = save(tmp_path / "parts/station.yaml", {
        "schema_version": 1, "root": {"name": "station", "children": [
            {"name": "robot", "assembly": "inner/robot.yaml",
             "overrides": {"eye": {"parameters": {"width": 240}}}},
        ]},
    })
    patch = {"overrides": {"robot/eye": {"parameters": {"width": 640}}}}
    station = load_assembly(outer, cameras=True, overrides=patch, factories=FACTORIES)
    assert station.children[0].children[0].parameters == {"width": 640, "height": 120}
    inactive = load_assembly(outer, cameras=False, overrides=patch, factories=FACTORIES)
    assert inactive.children[0].children == ()
    with pytest.raises(ValueError, match="Unknown overrides target 'robot/typo'") as error:
        load_assembly(outer, cameras=False, overrides={"overrides": {"robot/typo": {"pose": {}}}},
                      factories=FACTORIES)
    assert str(outer) in str(error.value)
    assert inner.is_file()


@pytest.mark.parametrize("patch, message", [
    ({"overrides": {"tool": {"name": "renamed"}}}, "Unknown overrides tool fields"),
    ({"overrides": {"tool": {"children": []}}}, "Unknown overrides tool fields"),
    ({"overrides": {"../tool": {}}}, "relative component path"),
    ({"overrides": {"/tool": {}}}, "relative component path"),
    ({"overrides": {"tool": None}}, "must be a mapping"),
    ({"overrides": []}, "must map relative names"),
    ({"connection_overrides": {"missing": {}}}, "Unknown connection_overrides target"),
    ({"connection_overrides": {"tool_mount": {"name": "renamed"}}}, "Unknown connection_overrides"),
])
def test_mistakes_fail_before_simulation(patch, message):
    with pytest.raises(ValueError, match=message):
        load_assembly(ARM, overrides=patch, factories=FACTORIES)


def test_connection_override_cannot_escape_assembly(tmp_path):
    path = save(tmp_path / "scene.yaml", {"schema_version": 1, "name": "world", "entities": [
        {"name": "arm", "assembly": str(ARM),
         "connection_overrides": {"tool_mount": {"parent": {"entity": "../other"}}}},
        {"name": "other", "factory": "box"},
    ]})
    with pytest.raises(ValueError, match="escapes its assembly"):
        load_world_spec(path, factories=FACTORIES)


def test_lift_default_robot_is_preserved_and_copy_can_customize_its_camera(tmp_path):
    from tools.create import create_project
    from shared.assembly.spec import BodyFrame, Pose, Robot

    target, _ = create_project("camera_project", "assembly_lift", destination=tmp_path)
    path = target / "scene.yaml"
    original = load_world_spec(path).index()["world/robot"][0]
    expected = Robot("robot", factory="franka_panda", pose_authored=False,
        frames=(BodyFrame("tcp", "panda_hand", Pose((0, 0, .1034))),))
    assert asdict(original) == asdict(expected)
    config = OmegaConf.load(path)
    arm = next(item for item in config.entities if item.name == "robot")
    arm.overrides = {"wrist_camera": {"parameters": {"width": 320, "height": 240}}}
    OmegaConf.save(config, path)
    world = load_world_spec(path, cameras=True)
    assert world.index()["world/robot/wrist_camera"][0].parameters == {"width": 320, "height": 240}
    assert world.index()["world/robot"][0].frames == expected.frames
    assert (target / "assemblies/arm.yaml").is_file()

