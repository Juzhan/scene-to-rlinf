import ast
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import ray
import torch
from omegaconf import OmegaConf

from run import ROOT, compose_project, execute, project_directories, recipe_paths


def migrated_projects():
    """Projects on the current assembly framework; unmigrated experiments are out of scope."""
    from shared.assembly.layout import load_scene_config

    for directory in project_directories():
        scene = directory / 'scene.yaml'
        if not scene.is_file() or not (directory / 'task.py').is_file():
            continue
        try:
            settings = load_scene_config(scene)
        except ValueError:
            continue
        if 'control' in settings:
            yield directory


def interface_report(name):
    """Measured native interface used by the maintained templates, or None."""
    from shared.assembly.interfaces import load_interface

    return load_interface(ROOT / 'projects' / name / 'scene.yaml')


class ProjectTests(unittest.TestCase):
    def test_smolvla_gui_requires_desktop_display(self):
        from projects.franka_stack_vla.rollout import main

        with patch.dict("os.environ", {"DISPLAY": ""}):
            with self.assertRaises(SystemExit) as error:
                main(["--gui"])
        self.assertEqual(error.exception.code, 2)

    @unittest.skip("未迁移到当前框架：旧工程保留作参考，不再对齐其训练入口")
    def test_smolvla_project_has_only_residual_registration_and_no_training_recipe(self):
        from projects.franka_stack_vla import project

        directory = ROOT / "projects/franka_stack_vla"
        self.assertEqual(recipe_paths(directory), ())
        self.assertIn(directory, tuple(project_directories()))
        with patch("rlinf.models.register_model") as register:
            project.register()
        self.assertEqual(register.call_count, 1)
        self.assertEqual(register.call_args.args[0], "smolvla_residual")
        with self.assertRaisesRegex(ValueError, "Not a training recipe"):
            compose_project("franka_stack_vla")

    def test_project_probe_routes_without_composing_training_recipe(self):
        import run

        module = SimpleNamespace(probe=Mock(return_value="probe-result"))
        with patch.object(sys, "argv", ["run.py", "probe", "franka_stack_vla", "--steps", "8"]), \
             patch.object(run, "import_module", return_value=module) as imported, \
             patch.object(run, "compose_project") as compose:
            self.assertEqual(run.main(), "probe-result")
        imported.assert_called_once_with("projects.franka_stack_vla.project")
        module.probe.assert_called_once_with(["--steps", "8"])
        compose.assert_not_called()

    def test_project_probe_rejects_missing_hook(self):
        import run

        with patch.object(sys, "argv", ["run.py", "probe", "franka_stack_vla"]), \
             patch.object(run, "import_module", return_value=SimpleNamespace()):
            with self.assertRaises(SystemExit) as error:
                run.main()
        self.assertEqual(error.exception.code, 2)

    def test_state_projects_choose_their_model_locally(self):
        from rlinf.models import _MODEL_REGISTRY, get_model

        common = OmegaConf.load(ROOT / "configs/state_ppo.yaml")
        self.assertTrue(OmegaConf.is_missing(common.actor.model, "model_type"))
        self.assertNotIn("model/mlp_policy@actor.model", common.defaults)
        projects = []
        for directory in migrated_projects():
            name = directory.name
            recipe = directory / "train.yaml"
            local = OmegaConf.load(recipe)
            if "state_ppo" not in local.get("defaults", []):
                continue
            with self.subTest(project=name):
                original_mlp = _MODEL_REGISTRY["mlp_policy"]
                cfg, _ = compose_project(name)
                self.assertEqual(cfg.actor.model.model_type, local.actor.model.model_type)
                if name == "assembly_go2":
                    from projects.assembly_go2 import project
                    from shared.integrations.rlinf.isaac import get_isaac_assembly_factory

                    scene = OmegaConf.load(recipe.with_name("scene.yaml"))
                    self.assertNotIn("env_cfg", cfg.env.train.init_params)
                    self.assertNotIn("preview_factory", cfg.env.train.init_params)
                    self.assertEqual(get_isaac_assembly_factory(cfg.env.train.init_params.id), project.make_environment)
                    self.assertEqual(next(entity for entity in scene.entities if entity.name == "robot").factory, "go2")
                    self.assertNotIn("_target_", recipe.with_name("scene.yaml").read_text())
                    self.assertNotIn("isaaclab.", recipe.with_name("scene.yaml").read_text())
                    self.assertNotIn("projects.go2", recipe.read_text())
                if name == "assembly_lift":
                    self.assertEqual(cfg.actor.model.model_type, "assembly_lift_policy")
                    self.assertIs(_MODEL_REGISTRY["mlp_policy"], original_mlp)
                    self.assertEqual(cfg.actor.model.action_dim, "auto")
                    # Object inputs: obs_dim stays null and field shapes come from inspect.
                    report = interface_report(name)
                    if report is None:
                        self.skipTest(f"{name}: run tools/inspect_scene.py {name} first")
                    from tools.inspect_scene import apply_dimensions
                    self.assertIsNone(cfg.actor.model.obs_dim)
                    apply_dimensions(cfg, report)
                    model = get_model(cfg.actor.model)
                    # PPO wraps the project model in the shared object adapter.
                    self.assertEqual(type(model.model).__module__, "projects.assembly_lift.policy")
                    observations = {key: torch.zeros(2, *shape)
                                    for key, shape in cfg.actor.model.observation_shapes.items()}
                    actions, _ = model.predict_action_batch(observations)
                    self.assertEqual(actions.shape, (2, 1, cfg.actor.model.action_dim))
            projects.append(name)
        self.assertIn("assembly_lift", projects)

    def test_assembly_projects_use_the_same_configuration_entry(self):
        from projects.assembly_go2 import project as go2
        from projects.assembly_lift import project as lift
        self.assertIs(type(go2), type(lift))
        self.assertEqual(go2.scene_config, ROOT / "projects/assembly_go2/scene.yaml")
        self.assertEqual(lift.scene_config, ROOT / "projects/assembly_lift/scene.yaml")

    def test_all_native_configs_and_policy_shapes_without_simulation(self):
        from rlinf.envs import get_env_cls
        from rlinf.models import get_model

        for directory in migrated_projects():
            name = directory.name
            for recipe in recipe_paths(directory):
                with self.subTest(project=name, recipe=recipe.stem):
                    try:
                        cfg, module = compose_project(name, recipe.stem)
                        self.assertIsNotNone(get_env_cls(cfg.env.train.env_type, cfg.env.train))
                    except ModuleNotFoundError as error:
                        if error.name not in ("gym", "franka_sim"):
                            raise
                        self.skipTest(f"Optional simulator dependency is unavailable: {error.name}")
                    from shared.assembly.project import Project
                    assembly = isinstance(getattr(module, "project", None), Project)
                    if assembly and any(cfg.actor.model.get(key) == "auto" for key in ("obs_dim", "action_dim")):
                        # Asset-resolved dimensions are checked by inspect/smoke;
                        # this test intentionally does not launch a simulator.
                        self.assertTrue(module.check(cfg)["actions"])
                        continue
                    if assembly or cfg.actor.model.model_type == "mlp_policy":
                        model = get_model(cfg.actor.model)
                        observations = {"states": torch.zeros(2, cfg.actor.model.obs_dim)}
                        policy_module = import_module(type(model).__module__)
                        for field in getattr(policy_module, "IMAGE_FIELDS", ()):
                            observations[field] = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
                        actions, _ = model.predict_action_batch(observations)
                        self.assertEqual(actions.shape, (2, 1, cfg.actor.model.action_dim))
                    elif cfg.actor.model.model_type == "cnn_policy":
                        self.assertGreater(cfg.actor.model.state_dim, 0)
                        self.assertEqual(cfg.actor.model.num_action_chunks, 1)
                    else:
                        self.fail(f"Unhandled policy interface: {cfg.actor.model.model_type}")

    def test_package_registration_loads_in_a_fresh_native_process(self):
        report = interface_report("assembly_lift")
        if report is None:
            self.skipTest("run tools/inspect_scene.py assembly_lift first")
        script = """
import os
import json
import sys
import torch
from run import compose_project, project_directories
from rlinf.models import _MODEL_REGISTRY, get_model
from rlinf.scheduler.cluster.utils import load_user_extension_module
from tools.inspect_scene import apply_dimensions

assert 'assembly_lift' in {path.name for path in project_directories()}
cfg, module = compose_project('assembly_lift')
assert module.__name__ == 'projects.assembly_lift'
assert os.environ['RLINF_EXT_MODULE'] == 'projects.assembly_lift'
del _MODEL_REGISTRY['assembly_lift_policy']
load_user_extension_module()
assert cfg.actor.model.obs_dim is None and cfg.actor.model.action_dim == 'auto'
apply_dimensions(cfg, json.loads(sys.argv[1]))
model = get_model(cfg.actor.model)
assert type(model.model).__module__ == 'projects.assembly_lift.policy'
observations = {key: torch.zeros(2, *shape) for key, shape in cfg.actor.model.observation_shapes.items()}
actions, _ = model.predict_action_batch(observations)
assert actions.shape == (2, 1, cfg.actor.model.action_dim)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, json.dumps(report)], cwd=ROOT,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_old_platform_or_training_imports(self):
        # Shared code and the maintained projects must not reach back into the
        # platforms the framework replaced.
        roots = [ROOT / "shared", *(directory for directory in migrated_projects())]
        for root in roots:
            for path in root.rglob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    modules = []
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    self.assertFalse(any(module.startswith(("robot_platform", "rl_training", "rsl_rl", "stable_baselines3")) for module in modules), str(path))
        self.assertFalse(any(module.startswith("robot_platform") for module in sys.modules))

    def test_project_generator_is_local_and_refuses_overwrite(self):
        from tools.create import create_project

        with tempfile.TemporaryDirectory() as directory:
            target, files = create_project("probe", dry_run=True, destination=directory)
            self.assertFalse(target.exists())
            target, files = create_project("probe", destination=directory)
            self.assertTrue((target / "policy.py").is_file())
            self.assertIn("build_planner", (target / "policy.py").read_text())
            self.assertIn("build_executor", (target / "policy.py").read_text())
            self.assertFalse((target / "env.py").exists())
            self.assertFalse((target / "controllers.py").exists())
            with self.assertRaises(FileExistsError):
                create_project("probe", destination=directory)
            with self.assertRaises(ValueError):
                create_project("../escape", destination=directory)
            cfg = OmegaConf.load(target / "train.yaml")
            # Copying a template namespaces its registrations for the new project.
            self.assertEqual(cfg.actor.model.model_type, "probe__planner_executor_policy")
            self.assertIsNone(cfg.actor.model.obs_dim)
            self.assertEqual(cfg.actor.model.action_dim, "auto")

    def test_generated_project_loads_in_a_fresh_native_process(self):
        from tools.create import create_project

        report = interface_report("planner_executor")
        if report is None:
            self.skipTest("run tools/inspect_scene.py planner_executor first")
        script = """
import os
import sys
import json
import projects
from run import ROOT, RLINF
from hydra import compose, initialize_config_dir
from rlinf.scheduler.cluster.utils import load_user_extension_module
from rlinf.models import get_model
import torch

projects.__path__.append(sys.argv[1])
os.environ['RLINF_EXT_MODULE'] = 'projects.probe'
load_user_extension_module()
with initialize_config_dir(version_base='1.3', config_dir=sys.argv[1] + '/probe'):
    cfg = compose(config_name='train', overrides=[
        f'hydra.searchpath=[file://{ROOT}/configs,file://{RLINF}/examples/embodiment/config]'
    ])
from tools.inspect_scene import apply_dimensions
apply_dimensions(cfg, json.loads(sys.argv[2]))
model = get_model(cfg.actor.model)
assert type(model.model).__module__ == 'projects.probe.policy'
observations = {key: torch.zeros(2, *shape) for key, shape in cfg.actor.model.observation_shapes.items()}
actions, _ = model.predict_action_batch(observations)
assert actions.shape == (2, 1, cfg.actor.model.action_dim)
assert cfg.env.train.init_params.id == 'probe/Workspace-Planner-Executor-v0'
assert not any(name.startswith('robot_platform') for name in sys.modules)
"""
        with tempfile.TemporaryDirectory() as directory:
            create_project("probe", destination=directory)
            result = subprocess.run(
                [sys.executable, "-c", script, directory, json.dumps(report)], cwd=ROOT,
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_official_train_and_eval_delegation(self):
        cfg, _ = compose_project("cartpole")
        for operation, module in (("train", "examples.embodiment.train_embodied_agent"),
                                  ("evaluate", "evaluations.eval_embodied_agent")):
            for error in (None, RuntimeError("native failure")):
                main = Mock(side_effect=error)
                with patch.dict(sys.modules, {module: SimpleNamespace(main=main)}), patch.object(ray, "shutdown") as shutdown:
                    candidate = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                    if error is None:
                        execute(candidate, operation)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "native failure"):
                            execute(candidate, operation)
                    main.assert_called_once_with(candidate)
                    shutdown.assert_called_once()
                    if operation == "evaluate":
                        self.assertTrue(candidate.runner.only_eval)
                        self.assertEqual(candidate.rollout.model.obs_dim, 4)

    def test_official_async_train_and_eval_delegation(self):
        cfg, _ = compose_project("cartpole")
        OmegaConf.update(cfg, "runner.async_training", True, force_add=True)
        for operation, module in (("train", "examples.embodiment.train_async"),
                                  ("evaluate", "evaluations.eval_embodied_agent")):
            for error in (None, RuntimeError("native failure")):
                main = Mock(side_effect=error)
                with patch.dict(sys.modules, {module: SimpleNamespace(main=main)}), patch.object(ray, "shutdown") as shutdown:
                    candidate = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                    if error is None:
                        execute(candidate, operation)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "native failure"):
                            execute(candidate, operation)
                    main.assert_called_once_with(candidate)
                    shutdown.assert_called_once()

    @unittest.skip("未迁移到当前框架：旧工程保留作参考，不再对齐其模型构建")
    def test_vla_builder_delegates_and_preflight_fails_before_loading(self):
        cfg, project = compose_project("franka_stack_vla")
        from projects.franka_stack_vla.policy import build_model

        with patch("rlinf.models.embodiment.gr00t.get_model") as factory:
            build_model(cfg.actor.model, torch.bfloat16)
            factory.assert_called_once_with(cfg.actor.model, torch.bfloat16)
        with patch("importlib.util.find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "runtime is not ready"):
                project.preflight(cfg)
