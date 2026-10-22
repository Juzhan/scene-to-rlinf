"""Automatic model dimensions use a measured native interface."""
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from omegaconf import OmegaConf
from tools.inspect_scene import apply_dimensions


def test_resolves_both_dimensions_and_existing_hydra_references():
    cfg = OmegaConf.create({"actor": {"model": {"obs_dim": "auto", "action_dim": "auto"}},
                            "env": {"dim": "${actor.model.obs_dim}"}})
    report = {"observations": {"states": {"shape": [17]}}, "action_dim": 6}
    apply_dimensions(cfg, report)
    assert cfg.env.dim == 17 and cfg.actor.model.action_dim == 6


def test_custom_image_shape_is_not_silently_flattened():
    cfg = OmegaConf.create({"actor": {"model": {"obs_dim": "auto", "action_dim": 6}}})
    with pytest.raises(ValueError, match="flat 'states'"):
        apply_dimensions(cfg, {"observations": {"states": {"shape": [64, 64, 3]}}, "action_dim": 6})


def test_explicit_custom_model_dimensions_are_preserved():
    cfg = OmegaConf.create({"actor": {"model": {"obs_dim": 9, "action_dim": "auto"}}})
    apply_dimensions(cfg, {"observations": {}, "action_dim": 3})
    assert cfg.actor.model.obs_dim == 9 and cfg.actor.model.action_dim == 3


def _scene(path, **updates):
    value = {"schema_version": 1, "name": "world", "entities": [],
             "observations": {"states": ["joint_pos", "joint_vel"]}, **updates}
    OmegaConf.save(OmegaConf.create(value), path)
    return path


@pytest.mark.parametrize("only_eval, selected", [(False, "train"), (True, "eval")])
def test_worker_measures_selected_split_and_closes_its_environment(tmp_path, monkeypatch, only_eval, selected):
    import tools.inspect_scene as inspection

    calls = []
    module = ModuleType("_interface_worker_test")
    environment = SimpleNamespace(
        native=object(), reset=lambda **options: calls.append(("reset", options)),
        close=lambda: calls.append(("environment_close",)))
    owner = SimpleNamespace(close=lambda: calls.append(("owner_close",)))

    def make_environment(parameters, count, seed, horizon, **options):
        calls.append(("create", dict(parameters), count, seed, horizon, options))
        return environment, owner

    module.project = SimpleNamespace(make_environment=make_environment)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    report = {"action_dim": 2, "articulations": {"robot": {"joint_names": ["left", "right"]}}}
    monkeypatch.setattr(inspection, "native_interface", lambda native: report)
    recipe, output = tmp_path / "recipe.yaml", tmp_path / "report.json"
    config = {"runner": {"only_eval": only_eval}, "env": {
        "train": {"init_params": {"scene_config": "TRAIN.yaml"}, "seed": 3, "max_episode_steps": 20},
        "eval": {"init_params": {"scene_config": "EVAL.yaml"}, "seed": 5, "max_episode_steps": 40}}}
    OmegaConf.save(OmegaConf.create(config), recipe)
    inspection.worker(SimpleNamespace(module=module.__name__, recipe=recipe, output=output))
    expected = config["env"][selected]
    assert calls == [
        ("create", expected["init_params"], 1, expected["seed"], expected["max_episode_steps"], {"headless": True}),
        ("reset", {"seed": expected["seed"]}), ("environment_close",), ("owner_close",)]
    assert json.loads(output.read_text()) == report


@pytest.mark.parametrize("only_eval, selected", [(False, "train"), (True, "eval")])
def test_probe_caches_interface_for_selected_scene_only(tmp_path, monkeypatch, only_eval, selected):
    import tools.inspect_scene as inspection
    from shared.assembly.interfaces import load_interface

    scenes = {name: _scene(tmp_path / f"{name}.yaml") for name in ("train", "eval")}
    config = OmegaConf.create({"runner": {"only_eval": only_eval}, "env": {
        name: {"init_params": {"scene_config": str(path)}} for name, path in scenes.items()}})
    module = SimpleNamespace(__name__="probe_test", project=SimpleNamespace(
        scene_config=scenes["train"], make_environment=lambda: None))
    report = {"action_dim": 2, "source_split": selected}

    def probe(command, **options):
        recipe = OmegaConf.load(command[command.index("--recipe") + 1])
        assert recipe.runner.only_eval is only_eval
        Path(command[command.index("--output") + 1]).write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(inspection.subprocess, "run", probe)
    assert inspection.inspect_configuration(config, module) == report
    assert module.interface_metadata == report
    assert load_interface(scenes[selected]) == report
    other = "train" if selected == "eval" else "eval"
    assert load_interface(scenes[other]) is None


@pytest.mark.parametrize("edit", ["scene", "asset"])
def test_probe_discards_metadata_if_scene_or_asset_changes_during_measurement(tmp_path, monkeypatch, edit):
    import tools.inspect_scene as inspection
    from shared.assembly.interfaces import load_interface

    scene = _scene(tmp_path / "scene.yaml")
    asset = tmp_path / "robot.usd"
    asset.write_text("joint_order = left,right\n")
    config = OmegaConf.create({"runner": {"only_eval": False},
                              "env": {"train": {"init_params": {"scene_config": str(scene)}}}})
    module = SimpleNamespace(__name__="probe_test", project=SimpleNamespace(
        scene_config=scene, make_environment=lambda: None))

    def probe(command, **options):
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "observations": {"states": {"terms": ["joint_pos", "joint_vel"]}}, "action_dim": 2}))
        # The native worker measured the old files before this concurrent edit.
        if edit == "scene":
            _scene(scene, observations={"states": ["joint_vel", "joint_pos"]})
        else:
            asset.write_text("joint_order = right,left\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(inspection.subprocess, "run", probe)
    with pytest.raises(ValueError, match="changed during interface inspection"):
        inspection.inspect_configuration(config, module)
    assert not hasattr(module, "interface_metadata")
    assert load_interface(scene) is None


@pytest.mark.parametrize("reference", ["local", "external_scene", "external_assembly"])
def test_changed_robot_assets_invalidate_measured_joint_metadata(tmp_path, reference):
    from shared.assembly.interfaces import load_interface, save_interface

    project = tmp_path / "project"
    project.mkdir()
    asset = (project if reference == "local" else tmp_path) / "robot.usd"
    asset.write_text("joint_order = left,right\n")
    if reference == "external_assembly":
        assembly = tmp_path / "arm.yaml"
        OmegaConf.save(OmegaConf.create({"schema_version": 1, "root": {
            "name": "robot", "factory": "custom_robot", "parameters": {"usd_path": "robot.usd"}}}), assembly)
        entities = [{"name": "robot", "assembly": "../arm.yaml"}]
    else:
        entities = [{"name": "robot", "factory": "custom_robot", "parameters": {
            "usd_path": "robot.usd" if reference == "local" else "../robot.usd"}}]
    scene = _scene(project / "scene.yaml", entities=entities)
    report = {"articulations": {"robot": {"joint_names": ["left", "right"]}}}
    save_interface(scene, report)
    assert load_interface(scene) == report
    asset.write_text("joint_order = right,left\n")
    assert load_interface(scene) is None
