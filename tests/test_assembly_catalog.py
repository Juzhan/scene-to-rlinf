"""Users can discover meaningful native terms without starting a simulator."""

import json
from pathlib import Path
import subprocess
import sys

from shared.assembly.catalog import describe_scene


ROOT = Path(__file__).resolve().parents[1]


def test_catalog_names_explain_real_observation_and_reward_meaning():
    report = describe_scene(ROOT / "projects/go2_stairs/scene.yaml")
    observations = {entry["id"]: entry for entry in report["observations"]}
    assert observations["actions"]["label"] == "最近一次策略动作"
    assert "原始策略输入" in observations["actions"]["description"]
    assert observations["joint_pos"]["dimension"] is None
    assert observations["joint_pos"]["implementation"].endswith(".joint_pos_rel")
    assert observations["height_scan"]["available"]
    rewards = {entry["id"]: entry for entry in report["rewards"]}
    assert rewards["dof_torques_l2"]["formula"] == "sum(applied_torque[selected_joints]^2)"
    assert "不是能量" in rewards["dof_torques_l2"]["description"]
    assert rewards["dof_torques_l2"]["overrides"]["weight"] == -.0002
    assert rewards["undesired_contacts"]["enabled"] is False
    action = next(entry for entry in report["actions"] if entry["id"] == "joint_position")
    assert action["targets"][0]["native_key"] == "robot"
    assert "robot" in action["parameters"]["asset_name"]["choices"]
    assert action["availability"] == "candidate" and action["runtime_required"]
    json.dumps(report, ensure_ascii=False)


def test_lift_target_is_explicit_three_dimensions_and_regions_do_not_become_observations():
    report = describe_scene(ROOT / "projects/assembly_lift/scene.yaml")
    target = next(entry for entry in report["observations"] if entry["id"] == "object.target.position")
    assert target["dimension"] == 3
    assert target["value"] == [.5, 0, .3]
    assert not any(entry["id"] == "region" for entry in report["observations"])


def test_missing_scene_dependencies_are_reported_and_custom_factories_stay_lazy(tmp_path):
    def body(strength=1):
        raise AssertionError("Catalog must not instantiate native asset")

    scene = {"name": "world", "schema_version": 1,
             "entities": [{"name": "arm", "kind": "robot", "factory": "custom_arm"}],
             "task": {"preset": "velocity"}, "observations": {"states": ["height_scan"]},
             "actions": {"arm": {"type": "joint_position", "asset_name": "arm", "joint_names": [".*"]}}}
    report = describe_scene(scene, directory=tmp_path, factories={"custom_arm": body}, asset_metadata={
        "custom_arm": {"label": "定制机械臂", "preview": {"path": "arm.usd"}}})
    scanner = next(entry for entry in report["observations"] if entry["id"] == "height_scan")
    assert scanner["available"] is False and scanner["missing"] == ["height_scanner"]
    asset = next(entry for entry in report["assets"] if entry["id"] == "custom_arm")
    assert asset["label"] == "定制机械臂" and asset["preview"] == {"path": "arm.usd"}
    assert asset["parameters"]["strength"]["default"] == 1
    assert report["actions"][0]["targets"][0]["native_key"] == "arm"


def test_catalog_fresh_process_imports_no_sdk_or_tensor_runtime():
    code = """
import json, sys
from shared.assembly.catalog import describe_scene
report = describe_scene('projects/assembly_go2/scene.yaml')
assert report['entities']
assert not any(name.split('.')[0] in ('isaaclab', 'isaacsim', 'omni', 'pxr', 'torch') for name in sys.modules)
print(json.dumps({'observations': len(report['observations'])}))
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["observations"] > 0


def test_entity_categories_and_observation_bindings_follow_actual_targets(tmp_path):
    scene = {"schema_version": 1, "name": "world", "entities": [
        {"name": "robot", "factory": "go2", "kind": "robot"},
        {"name": "arm", "factory": "franka_panda", "kind": "entity"},
        {"name": "floor", "factory": "ground_plane"}],
        "task": {"preset": "velocity"},
        "observations": {"states": {"terms": ["joint_pos", "actions"], "overrides": {
            "joint_pos": {"params": {"asset_cfg": {"name": "arm"}}}}}},
        "actions": {"legs": {"type": "joint_position", "asset_name": "robot", "joint_names": [".*"]}}}
    report = describe_scene(scene, directory=tmp_path)
    entities = {entry["path"]: entry for entry in report["entities"]}
    assert entities["world/arm"]["category"] == "robot"  # capability, not YAML role
    assert entities["world/floor"]["category"] == "terrain"
    terms = {entry["id"]: entry for entry in report["observations"]}
    joint = terms["joint_pos"]["source_groups"]["states"]
    assert joint["group"] == "world/arm" and joint["available"]
    assert joint["sources"][0]["path"] == "world/arm"
    assert terms["actions"]["source_groups"]["states"]["group"] == "actions"
    assert terms["velocity_commands"]["source_groups"]["states"]["group"] == "commands"
    scene["observations"]["states"]["overrides"]["joint_pos"]["params"]["asset_cfg"]["name"] = "missing_arm"
    missing = next(entry for entry in describe_scene(scene, directory=tmp_path)["observations"] if entry["id"] == "joint_pos")
    assert missing["source_groups"]["states"]["missing"] == ["missing_arm"]


def test_relative_observations_keep_both_entity_sources():
    report = describe_scene(ROOT / "projects/assembly_lift/scene.yaml")
    relation = next(entry for entry in report["observations"] if entry["id"] == "object.position")
    source = relation["source_groups"]["default"]
    assert source["group"] == "world/object"
    assert {item["path"] for item in source["sources"]} == {"world/object"}


def test_catalog_includes_only_current_file_interface_cache(tmp_path):
    from omegaconf import OmegaConf
    from shared.assembly.interfaces import save_interface

    scene = {"schema_version": 1, "name": "world", "entities": [
        {"name": "robot", "factory": "go2", "kind": "robot"}],
        "observations": {"states": ["joint_pos"]}, "task": {"preset": "velocity"},
        "actions": {"legs": {"type": "joint_position", "asset_name": "robot", "joint_names": [".*"]}}}
    path = tmp_path / "scene.yaml"
    OmegaConf.save(OmegaConf.create(scene), path)
    measured = {"articulations": {"robot": {"joint_names": ["actual_joint"]}}, "action_dim": 1}
    save_interface(path, measured)
    catalog = describe_scene(path)
    assert catalog["resolved_interface"] == measured
    assert catalog["actions"][0]["targets"][0]["joint_names"] == ["actual_joint"]
    assert describe_scene(scene, directory=tmp_path)["resolved_interface"] is None
    path.write_text(path.read_text() + "\n# edited\n")
    assert describe_scene(path)["resolved_interface"] is None


def test_camera_image_shape_and_empty_component_selection_while_editing():
    import yaml
    settings = yaml.safe_load((ROOT / "projects/piper_pick/scene.yaml").read_text())
    camera = next(item for item in settings["entities"] if item["name"] == "wrist_camera")
    camera.setdefault("parameters", {}).update(width=64, height=48)
    settings["control"]["components"]["planner"]["observations"] = []  # Just deleted its only entity.
    report = describe_scene(settings, directory=ROOT / "projects/piper_pick")
    rgb = next(entry for entry in report["observations"] if entry["id"] == "wrist_camera.rgb")
    assert list(rgb["information"]["shape"]) == [48, 64, 3], rgb["information"]


def test_camera_outputs_follow_data_types_and_parameters_are_typed():
    import pytest
    import yaml
    from shared.assembly.assets import check_parameters
    from shared.backends.isaac.assets import FACTORIES
    settings = yaml.safe_load((ROOT / "projects/piper_pick/scene.yaml").read_text())
    camera = next(item for item in settings["entities"] if item["name"] == "wrist_camera")
    camera["parameters"] = {"width": 64, "height": 48, "data_types": ["distance_to_image_plane", "normals"]}
    report = describe_scene(settings, directory=ROOT / "projects/piper_pick")
    images = {entry["id"]: entry for entry in report["observations"] if entry["id"].startswith("wrist_camera.")
              and entry["information"]["kind"] == "image"}
    assert set(images) == {"wrist_camera.distance_to_image_plane", "wrist_camera.normals"}
    depth = images["wrist_camera.distance_to_image_plane"]["information"]
    assert list(depth["shape"]) == [48, 64, 1] and depth["dtype"] == "float" and depth["unit"] == "m"
    assert images["wrist_camera.normals"]["profile"] == "camera__wrist_camera"
    fields = next(entry for entry in report["assets"] if entry["id"] == "camera")["parameters"]
    assert fields["width"]["type"] == "integer" and fields["focal_length"]["type"] == "number"
    assert fields["data_types"]["type"] == "choices" and "distance_to_camera" in fields["data_types"]["options"]
    assert fields["depth_clipping"]["type"] == "choice" and fields["clipping_range"]["length"] == 2
    box = next(entry for entry in report["assets"] if entry["id"] == "box")["parameters"]
    assert box["size"]["type"] == "vector" and box["mass"]["nullable"]
    for bad in ({"width": 0}, {"data_types": ["rgb", "rgb"]}, {"data_types": ["depth"]},
                {"clipping_range": [1.0, 0.5]}, {"depth_clipping": "far"}, {"lens": 3}):
        with pytest.raises((TypeError, ValueError)):
            check_parameters(FACTORIES["camera"], bad)
    with pytest.raises(ValueError, match="mass"):
        check_parameters(FACTORIES["box"], {"mass": -1})


def test_selected_image_must_be_rendered_by_its_camera(tmp_path):
    import pytest
    import shutil
    from shared.assembly.editing import SceneDocument, apply_edits
    project = tmp_path / "piper_pick"
    shutil.copytree(ROOT / "projects/piper_pick", project, ignore=shutil.ignore_patterns(".cache", "__pycache__"))
    scene = project / "scene.yaml"
    document = SceneDocument(scene)
    terms = document.data["control"]["components"]["executor"]["observations"] + ["wrist_camera.distance_to_camera"]
    with pytest.raises(ValueError, match="data_types"):
        apply_edits(scene, {"revision": document.revision, "observations": {"executor": terms}})
    revision = SceneDocument(scene).revision
    apply_edits(scene, {"revision": revision, "observations": {"executor": terms},
                        "entities": [{"path": "world/wrist_camera",
                                      "changes": {"parameters": {"data_types": ["rgb", "distance_to_camera"]}}}]})
    saved = SceneDocument(scene)
    assert saved.data["application"]["enable_cameras"] is True
    assert saved.entities["world/wrist_camera"]["parameters"]["data_types"] == ["rgb", "distance_to_camera"]
