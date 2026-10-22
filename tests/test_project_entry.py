"""The project entry point stays usable without importing a simulator."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

import run
from tools.create import create_project


def test_recipe_discovery_uses_contents_not_scene_filename():
    assert {path.stem for path in run.recipe_paths(run.ROOT / "projects/assembly_lift")} == {"train"}
    assert {path.stem for path in run.recipe_paths(run.ROOT / "projects/assembly_go2")} == {"train"}
    assert "scene" in {path.stem for path in run.recipe_paths(run.ROOT / "projects/piper_throw")}
    assert "course" in {path.stem for path in run.recipe_paths(run.ROOT / "projects/my_robot")}
    with pytest.raises(ValueError, match="Not a training recipe"):
        run.compose_project("assembly_lift", "scene")


def test_explicit_recipe_folder_and_legacy_defaults(tmp_path, monkeypatch):
    project = tmp_path / "projects/probe"
    (project / "recipes").mkdir(parents=True)
    (tmp_path / "configs").mkdir()
    (project / "__init__.py").write_text("")
    (project / "recipes/train.yaml").write_text(
        "runner: {save_interval: 1}\nenv: {train: {}}\nactor: {}\n"
    )
    (project / "scene.yaml").write_text("entities: {robot: {}}\n")
    module = SimpleNamespace(register=Mock())
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "import_module", lambda _: module)
    assert project in run.project_directories()
    assert run.recipe_paths(project) == (project / "recipes/train.yaml",)
    cfg, loaded = run.compose_project("probe")
    assert loaded is module and cfg.runner.save_interval == 1
    module.register.assert_called_once()
    (project / "train.yaml").write_text("runner: {}\nenv: {}\nactor: {}\n")
    with pytest.raises(ValueError, match="Duplicate recipe names"):
        run.recipe_paths(project)


def test_defaults_cycles_are_not_mistaken_for_recipes(tmp_path):
    (tmp_path / "one.yaml").write_text("defaults: [two]\n")
    (tmp_path / "two.yaml").write_text("defaults: [one]\n")
    assert run.recipe_paths(tmp_path) == ()


def test_test_command_delegates_to_pytest(monkeypatch):
    pytest_main = Mock(return_value=0)
    monkeypatch.setattr(pytest, "main", pytest_main)
    monkeypatch.setattr(sys, "argv", ["run.py", "test", "-q", "-k", "assembly"])
    with pytest.raises(SystemExit) as exit_info:
        run.main()
    assert exit_info.value.code == 0
    pytest_main.assert_called_once_with(
        [str(run.ROOT / "tests"), "--import-mode=importlib", "-q", "-k", "assembly"]
    )


def test_check_runs_project_static_hook(monkeypatch, capsys):
    import rlinf.envs

    cfg = OmegaConf.create({"runner": {"only_eval": False}, "env": {"train": {"env_type": "probe"}}})
    module = SimpleNamespace(check=Mock(return_value={"entities": 3}))
    monkeypatch.setattr(run, "compose_project", lambda *_: (cfg, module))
    monkeypatch.setattr(rlinf.envs, "get_env_cls", lambda *_: object)
    monkeypatch.setattr(sys, "argv", ["run.py", "check", "probe"])
    run.main()
    module.check.assert_called_once_with(cfg)
    assert '"entities": 3' in capsys.readouterr().out


@pytest.mark.parametrize("template", ["assembly_lift", "assembly_go2"])
def test_generated_assembly_coexists_with_template(tmp_path, template):
    from shared.assembly.interfaces import load_interface

    report = load_interface(run.ROOT / "projects" / template / "scene.yaml")
    if report is None:
        pytest.skip(f"run tools/inspect_scene.py {template} first")
    target, _ = create_project("assembly_copy", template, destination=tmp_path)
    (target / "assets.py").write_text(
        "def local_fixture(width=0.2):\n"
        "    raise AssertionError('Static check must not instantiate native assets')\n"
        "FACTORIES = {'local_fixture': local_fixture}\n"
    )
    scene = OmegaConf.load(target / "scene.yaml")
    fixture = {"factory": "local_fixture", "parameters": {"width": 0.3}}
    scene.entities.append({"name": "fixture", **fixture})
    if template == "assembly_lift":
        scene.application.enable_cameras = True
        arm = next(entity for entity in scene.entities if entity.name == "robot")
        arm.overrides = {"wrist_camera": {"parameters": {"width": 320, "height": 240}}}
    OmegaConf.save(scene, target / "scene.yaml")
    original = OmegaConf.load(run.ROOT / "projects" / template / "train.yaml")
    copied = OmegaConf.load(target / "train.yaml")
    assert copied.env.train.init_params.id != original.env.train.init_params.id
    assert copied.actor.model.model_type != original.actor.model.model_type
    script = """
import sys
import json
import projects
from importlib import import_module
from hydra import compose, initialize_config_dir
from run import ROOT, RLINF, compose_project
from rlinf.models import _MODEL_REGISTRY, get_model
from shared.integrations.rlinf.isaac import get_isaac_assembly_factory
from tools.inspect_scene import apply_dimensions
import torch

projects.__path__.append(sys.argv[1])
original, _ = compose_project(sys.argv[2])
model_factory = _MODEL_REGISTRY[original.actor.model.model_type]
environment_factory = get_isaac_assembly_factory(original.env.train.init_params.id)
copied_module = import_module('projects.assembly_copy')
copied_module.register()
copied_module.register()
with initialize_config_dir(version_base='1.3', config_dir=sys.argv[1] + '/assembly_copy'):
    copied = compose(config_name='train', overrides=[
        f'hydra.searchpath=[file://{ROOT}/configs,file://{RLINF}/examples/embodiment/config]'
    ])
assert _MODEL_REGISTRY[original.actor.model.model_type] is model_factory
assert get_isaac_assembly_factory(original.env.train.init_params.id) is environment_factory
assert get_isaac_assembly_factory(copied.env.train.init_params.id) is not environment_factory
report = copied_module.check(copied)
assert any(name.endswith('fixture') for name in report['entities'])
if sys.argv[2] == 'assembly_lift':
    assert report['enable_cameras'] is True
    camera = next(item for item in report['layout'] if item['path'].endswith('/wrist_camera'))
    assert camera['parameters'] == {'width': 320, 'height': 240}
    assert camera['mount'] == {'entity': 'world/robot', 'body': 'panda_hand'}
# Known CPU fixtures retain the independent registration/network check without
# asking the static project test to start an actual simulator.
assert copied.actor.model.obs_dim is None and copied.actor.model.action_dim == 'auto'
apply_dimensions(copied, json.loads(sys.argv[3]))
model = get_model(copied.actor.model)
observations = {key: torch.zeros(2, *shape) for key, shape in copied.actor.model.observation_shapes.items()}
actions, _ = model.predict_action_batch(observations)
assert actions.shape == (2, 1, copied.actor.model.action_dim)
assert not any(name.startswith('isaaclab.') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), template, json.dumps(report)],
                            cwd=run.ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_ids_and_external_asset_references_are_preserved(tmp_path):
    from tools.create import _rewrite_config_values, _rewrite_python_strings

    target, _ = create_project("cartpole_copy", "cartpole", destination=tmp_path)
    cfg = OmegaConf.load(target / "train.yaml")
    assert cfg.env.train.init_params.id == "Workspace-Cartpole-v0"
    assert cfg.env.train.init_params.env_cfg == "projects.cartpole_copy.env:BalanceEnvCfg"
    names = {"Workspace-Assembly-Lift-State-v0": "copy/Workspace-Assembly-Lift-State-v0"}
    url = "https://assets.example/Workspace-Assembly-Lift-State-v0.usd"
    assert _rewrite_config_values({"asset": url}, names) == {"asset": url}
    text = f"asset = {url!r}\ntask_id = 'Workspace-Assembly-Lift-State-v0'\n"
    rewritten = _rewrite_python_strings(text, names)
    assert f"asset = {url!r}" in rewritten
    assert "task_id = 'copy/Workspace-Assembly-Lift-State-v0'" in rewritten


def test_generator_rejects_ambiguous_assembly_registration(tmp_path):
    from tools.create import _assembly_registration_names

    (tmp_path / "__init__.py").write_text(
        "def register():\n"
        "    register_isaac_assembly_env('My-Task-v0', factory=make_environment)\n"
        "    register_model(compute_name(), build_model)\n"
    )
    with pytest.raises(ValueError, match="explicit string"):
        _assembly_registration_names(tmp_path, "copy")
