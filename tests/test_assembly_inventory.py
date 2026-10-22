"""Scene inspection explains authored instances without loading a simulator."""

import json
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf

from shared.assembly.project import check_project


def _scene(path, data):
    OmegaConf.save(OmegaConf.create(data), path)
    return path


def _config():
    environment = SimpleNamespace(init_params={})
    return SimpleNamespace(runner=SimpleNamespace(only_eval=False),
                           env=SimpleNamespace(train=environment, eval=environment))


def _no_asset_construction(**parameters):
    raise AssertionError("Static inspection must not construct native assets or presets")


def test_reused_assembly_inventory_distinguishes_layout_mounts_and_connections(tmp_path):
    _scene(tmp_path / "arm.yaml", {
        "schema_version": 1,
        "root": {
            "name": "arm", "kind": "robot", "factory": "robot",
            "children": [
                {"name": "tool", "factory": "tool", "pose": {"position": [0, 0, 9]}},
                {"name": "camera", "kind": "sensor", "factory": "camera", "camera_only": True,
                 "parameters": {"width": 320, "height": 240},
                 "pose": {"position": [0.1, 0, 0]},
                 "mount": {"entity": "..", "body": "wrist"}},
            ],
            "connections": [{"name": "tool_mount", "child": {"entity": "./tool"},
                             "parent": {"entity": ".", "body": "wrist"},
                             "parent_pose": {"position": [0, 0, 0.05]}}],
        },
    })
    path = _scene(tmp_path / "scene.yaml", {
        "schema_version": 1, "name": "world", "application": {"enable_cameras": True},
        "entities": [
            {"name": "left", "assembly": "arm.yaml", "pose": {"position": [1, 0, 0]},
             "parameters": {"strength": 1}},
            {"name": "right", "assembly": "arm.yaml", "pose": {"position": [-1, 0, 0]},
             "parameters": {"strength": 2}},
        ],
        "observations": {"states": ["joint_pos"]},
        "actions": {"arm": {"type": "joint_position", "asset_name": "left", "joint_names": [".*"]}},
        "task": {"preset": "velocity"},
    })
    before = set(sys.modules)
    report = check_project(_config(), scene_config=path,
                           factories={key: _no_asset_construction for key in ("robot", "tool", "camera")})
    report = json.loads(json.dumps(report))
    layout = {entry["path"]: entry for entry in report["layout"]}
    assert report["entities"] == list(layout)
    assert layout["world/left"]["kind"] == "robot"
    assert layout["world/left"]["parameters"] == {"strength": 1}
    assert layout["world/right"]["parameters"] == {"strength": 2}
    assert layout["world/left"]["local_pose"]["position"] == [1, 0, 0]
    assert layout["world/left/camera"]["local_pose"]["position"] == [0.1, 0, 0]
    assert layout["world/left/camera"]["parameters"] == {"width": 320, "height": 240}
    assert layout["world/left/camera"]["mount"] == {"entity": "world/left", "body": "wrist"}
    assert layout["world/right/camera"]["mount"] == {"entity": "world/right", "body": "wrist"}
    assert report["connections"] == len(report["physical_connections"]) == 2
    joint = report["physical_connections"][0]
    assert joint["parent"] == {"entity": "world/left", "body": "wrist"}
    assert joint["child"] == {"entity": "world/left/tool", "body": ""}
    assert joint["parent_pose"]["position"] == [0, 0, 0.05]
    assert layout["world/left/tool"]["local_pose"]["position"] == [0, 0, 9]
    assert "not measured world poses" in report["layout_note"]
    assert not any(name.startswith(("isaaclab", "pxr", "omni")) for name in set(sys.modules) - before)


def test_unified_inventory_preserves_scopes_mesh_and_mount_references(tmp_path):
    path = _scene(tmp_path / "scene.yaml", {
        "schema_version": 1, "name": "world",
        "entities": [
            {"name":"terrain", "prim_name":"ground", "factory":"local_ground", "scope":"global", "parameters":{"size":20}},
            {"name":"robot", "factory":"local_robot", "pose":{"position":[1,2,.5]}},
            {"name":"scanner", "kind":"sensor", "factory":"local_sensor", "mount":{"entity":"world/robot", "body":"base"},
             "meshes":["world/terrain"], "parameters":{"update_period":.1}},
        ],
        "observations":{"states":["joint_pos"]},
        "actions":{"joints":{"type":"joint_position", "asset_name":"robot", "joint_names":[".*"]}},
        "task":{"preset":"velocity"},
    })
    report = check_project(_config(), scene_config=path, factories={
        key:_no_asset_construction for key in ("local_ground", "local_robot", "local_sensor")})
    layout = {entry["path"]:entry for entry in report["layout"]}
    assert layout["world/terrain"]["scope"] == "global"
    assert layout["world/terrain"]["prim_name"] == "ground"
    assert layout["world/robot"]["local_pose"]["position"] == (1,2,.5)
    assert layout["world/scanner"]["mount"] == {"entity":"world/robot", "body":"base"}
    assert layout["world/scanner"]["meshes"] == ["world/terrain"]
    assert report["physical_connections"] == []
