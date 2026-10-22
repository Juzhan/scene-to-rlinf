"""A single project authoring path, without constructing any SDK assets."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from shared.assembly.project import prepare_scene, project_registry

ROOT = Path(__file__).resolve().parents[1]


def write_scene(path, data):
    OmegaConf.save(OmegaConf.create(data), path)
    return path


def base_scene():
    return {"schema_version": 1, "name": "world", "entities": [],
            "observations": {"states": ["joint_pos"]},
            "actions": {"joints": {"type": "joint_position", "asset_name": "robot", "joint_names": [".*"]}},
            "task": {"preset": "velocity"}}


def test_project_assets_and_terms_are_local_lazy_and_share_camera_selection(tmp_path, monkeypatch):
    package = tmp_path / "local_assembly_project"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "assets.py").write_text("def custom_cube(size=0.1):\n    raise AssertionError('No SDK in check')\ndef camera(width=64):\n    raise AssertionError('No SDK in check')\nFACTORIES = {'custom_cube':custom_cube, 'camera':camera}\n")
    (package / "terms.py").write_text("def image():\n    raise AssertionError('No SDK in check')\nTERMS = {'observations':{'image':image}}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    scene = base_scene()
    scene["entities"] = [{"name":"cube", "factory":"custom_cube"},
                         {"name":"camera", "kind":"sensor", "factory":"camera", "camera_only":True}]
    path = write_scene(package / "scene.yaml", scene)
    try:
        options = dict(scene_config=path, project_package=package.name)
        state = prepare_scene({}, **options)
        assert "world/camera" not in state.description.index()
        vision = prepare_scene({}, enable_cameras=True, **options)
        assert "world/camera" in vision.description.index()
        assert "image" in vision.terms["observations"]
        from shared.backends.isaac.assets import FACTORIES
        assert "custom_cube" not in FACTORIES
        scene["entities"][0]["parameters"] = {"misspelled":1}
        write_scene(path, scene)
        with pytest.raises(ValueError, match="world/cube.*misspelled") as error:
            prepare_scene({}, **options)
        assert str(path) in str(error.value)
    finally:
        for suffix in (".assets", ".terms", ""):
            sys.modules.pop(package.name + suffix, None)


@pytest.mark.parametrize("name", ["assembly_lift", "assembly_go2"])
def test_both_projects_have_one_entry_and_no_user_environment_or_noop_components(name):
    from run import compose_project
    from shared.assembly.project import Project
    before = set(sys.modules)
    cfg, module = compose_project(name)
    report = module.check(cfg)
    assert isinstance(module.project, Project)
    assert report["scene_config"] == str(ROOT / "projects" / name / "scene.yaml")
    assert report["entities"] and report["observations"]["policy"]
    assert report["actions"] and report["task"] == "task.py"
    assert report["task_evaluation"] is None  # Python Task has no legacy formula configuration.
    assert "scene_kind" not in report
    for filename in ("env.py", "action.py"):
        assert not (ROOT / "projects" / name / filename).exists()
    assert not any(name.startswith("isaaclab.") for name in set(sys.modules) - before)


def test_invalid_term_is_rejected_before_application_launch(tmp_path, monkeypatch):
    from shared.backends.isaac import environment
    scene = base_scene()
    scene["observations"] = {"states":["misspelled"]}
    path = write_scene(tmp_path / "scene.yaml", scene)
    monkeypatch.setattr(environment, "launch_application", lambda *_: pytest.fail("Launched before validation"))
    with pytest.raises(ValueError, match="unknown terms.*misspelled"):
        environment.make_environment({}, 2, 0, 8, scene_config=path)


def test_custom_task_terms_can_be_checked_without_constructing_a_preset(tmp_path):
    def native_factory():
        pytest.fail("check must not construct native config")
    scene = base_scene()
    scene["task"] = {"preset":"custom"}
    scene["observations"] = {"states":["local_measurement"]}
    path = write_scene(tmp_path / "scene.yaml", scene)
    prepared = prepare_scene({}, scene_config=path, tasks={"custom":native_factory})
    assert prepared.tasks["custom"] is native_factory
    with pytest.raises(ValueError, match="Unknown task preset"):
        prepare_scene({}, scene_config=path)


@pytest.mark.parametrize("terms", [None, {}, SimpleNamespace()])
def test_custom_manager_preset_gets_native_time_limit(monkeypatch, terms):
    from shared.backends.isaac.environment import _ensure_native_horizon
    timeout = lambda env: None
    monkeypatch.setitem(sys.modules, "isaaclab.envs", SimpleNamespace(mdp=SimpleNamespace(time_out=timeout)))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(TerminationTermCfg=SimpleNamespace))
    config = SimpleNamespace(terminations=terms)
    _ensure_native_horizon(config)
    entries = config.terminations if isinstance(config.terminations, dict) else vars(config.terminations)
    term = entries["workspace_time_limit"]
    assert term.func is timeout and term.time_out is True
    _ensure_native_horizon(config)
    assert entries["workspace_time_limit"] is term


def test_native_time_limit_preserves_existing_terms_and_rejects_reserved_collision(monkeypatch):
    from shared.backends.isaac.environment import _ensure_native_horizon
    timeout = lambda env: None
    monkeypatch.setitem(sys.modules, "isaaclab.envs", SimpleNamespace(mdp=SimpleNamespace(time_out=timeout)))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(TerminationTermCfg=SimpleNamespace))
    term = SimpleNamespace(func=timeout, time_out=True)
    config = SimpleNamespace(terminations={"time_out": term, "fallen": None})
    _ensure_native_horizon(config)
    assert config.terminations == {"time_out": term, "fallen": None}
    config.terminations = {"workspace_time_limit": SimpleNamespace(func=lambda env: None)}
    with pytest.raises(ValueError, match="reserved"):
        _ensure_native_horizon(config)
