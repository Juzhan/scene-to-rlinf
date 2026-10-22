from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from shared.isaac_state import IsaacStateEnv

ROOT = Path(__file__).resolve().parents[1]


class StateContractTests(unittest.TestCase):
    def test_native_chunk_preserves_terminal_states_without_second_reset(self):
        environment = object.__new__(IsaacStateEnv)
        environment.num_envs = 2
        environment.device = torch.device("cpu")
        environment.cfg = OmegaConf.create({"init_params": {"obs_dim": 4, "action_dim": 1}})
        environment.auto_reset = True
        environment.ignore_terminations = False
        environment._elapsed_steps = torch.zeros(2, dtype=torch.int32)
        environment.prev_step_reward = torch.zeros(2)
        environment._init_metrics()
        terminal_states = torch.full((2, 4), 9.0)
        environment.env = SimpleNamespace(
            step=Mock(return_value=({"policy": torch.zeros(2, 4)}, torch.tensor([1.0, 2.0]),
                torch.tensor([True, False]), torch.tensor([False, True]), {"terminal_states": terminal_states})),
            reset=Mock(),
        )
        observations, rewards, terminated, truncated, infos = environment.chunk_step(torch.zeros(2, 1, 1))
        torch.testing.assert_close(infos[-1]["final_observation"]["states"], terminal_states)
        self.assertEqual(float(observations[-1]["states"].sum()), 0)
        self.assertEqual(infos[-1]["final_info"]["episode"]["episode_len"].tolist(), [1, 1])
        self.assertEqual(environment.returns.tolist(), [0.0, 0.0])
        self.assertEqual(terminated[:, 0].tolist(), [True, False])
        self.assertEqual(truncated[:, 0].tolist(), [False, True])
        environment.env.reset.assert_not_called()
        with self.assertRaisesRegex(ValueError, "own adapter"):
            environment.chunk_step(torch.zeros(2, 2, 1))

    def test_preview_options_and_observation_contract(self):
        from tools.preview import checked_states, parse_options

        options, overrides = parse_options(["--steps", "0", "env.train.seed=17"], "view")
        self.assertEqual(options.actions, "zero")
        self.assertEqual(overrides, ["env.train.seed=17"])
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parse_options(["--steps", "0"], "record")
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            parse_options(["--random-scale", "nan"], "view")
        with self.assertRaisesRegex(ValueError, "finite"):
            checked_states({"policy": torch.full((1, 4), float("nan"))}, 1, 4)

    def test_zero_preview_records_and_closes_without_loading_a_model(self):
        from tools.preview import parse_options, preview

        options, _ = parse_options(["--num-envs", "2", "--steps", "4"], "record")
        observations = {"policy": torch.zeros(2, 4)}
        frame = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        environment = SimpleNamespace(
            device="cpu", step_dt=1 / 60, action_manager=SimpleNamespace(total_action_dim=1),
            sim=SimpleNamespace(is_stopped=Mock(return_value=False), is_playing=Mock(return_value=True),
                                render=Mock()),
            reset=Mock(return_value=(observations, {})), render=Mock(return_value=frame), close=Mock(),
            step=Mock(return_value=(observations, torch.ones(2), torch.zeros(2, dtype=torch.bool),
                                    torch.ones(2, dtype=torch.bool), {})),
        )
        owner = SimpleNamespace(application=SimpleNamespace(is_running=Mock(return_value=True)))
        with tempfile.TemporaryDirectory() as directory:
            cfg = OmegaConf.create({"env": {"train": {"env_type": "isaaclab", "seed": 42,
                "max_episode_steps": 8, "init_params": {"id": "test", "env_cfg": "test:Config",
                "obs_dim": 4, "action_dim": 1}}}, "runner": {"logger": {"log_path": directory}}})
            with patch("tools.preview.make_simulation", return_value=(environment, owner)) as factory, \
                 patch("rlinf.models.get_model") as model, patch("imageio.v2.get_writer") as writer, \
                 patch("imageio.v2.imwrite"), patch("builtins.print"):
                report = preview(cfg, options, recording=True)
                model.assert_not_called()
                self.assertTrue(factory.call_args.kwargs["headless"])
                self.assertTrue(factory.call_args.kwargs["enable_cameras"])
                self.assertEqual(writer.return_value.append_data.call_count, 4)
                writer.return_value.close.assert_called_once()
                environment.sim.render.assert_called_once()  # The first frame is already nonblank.
            environment.reset.assert_called_once()
            environment.close.assert_called_once()
            self.assertEqual(report["steps"], 4)
            self.assertEqual(report["resets"], 8)
            self.assertTrue(report["environment_closed"])
            self.assertFalse(report["training"])
            for call in environment.step.call_args_list:
                torch.testing.assert_close(call.args[0], torch.zeros(2, 1))
            self.assertEqual(len(list(Path(directory).rglob("report.json"))), 1)

    def test_registered_assembly_preview_needs_no_factory_configuration(self):
        from run import compose_project
        from shared.backends.isaac.environment import Environment
        from shared.integrations.rlinf.isaac import _ENVIRONMENT_FACTORIES
        from tools.preview import parse_options, preview

        cfg, _ = compose_project("assembly_go2")
        from tools.inspect_scene import apply_dimensions
        from shared.assembly.interfaces import load_interface
        report = load_interface(ROOT / "projects/assembly_go2/scene.yaml")
        if report is None:
            self.skipTest("run tools/inspect_scene.py assembly_go2 first")
        # Object inputs keep obs_dim null; named fields carry the measured shapes.
        apply_dimensions(cfg, report)
        self.assertNotIn("preview_factory", cfg.env.train.init_params)
        options, _ = parse_options(["--steps", "2"], "view")
        observations = {key: torch.zeros(1, *shape)
                        for key, shape in cfg.actor.model.observation_shapes.items()}
        native = SimpleNamespace(
            num_envs=1, device="cpu", step_dt=0.02,
            action_manager=SimpleNamespace(total_action_dim=cfg.env.train.init_params.action_dim),
            sim=SimpleNamespace(is_stopped=Mock(return_value=False), is_playing=Mock(return_value=True)),
            reset=Mock(return_value=(observations, {})), close=Mock(),
            step=Mock(return_value=(observations, torch.ones(1), torch.zeros(1, dtype=torch.bool),
                                   torch.ones(1, dtype=torch.bool), {"autoreset": True})),
        )
        environment = Environment(native)
        owner = SimpleNamespace(application=SimpleNamespace(is_running=Mock(return_value=True)))
        window_module = SimpleNamespace(get_default_app_window=Mock())
        with tempfile.TemporaryDirectory() as directory:
            cfg.runner.logger.log_path = directory
            constructor = Mock(return_value=(environment, owner))
            with patch.dict(_ENVIRONMENT_FACTORIES, {cfg.env.train.init_params.id: constructor}), \
                 patch("tools.preview.make_simulation") as fallback, \
                 patch.dict("os.environ", {"DISPLAY": ":test"}), \
                 patch.dict("sys.modules", {"omni.appwindow": window_module}), patch("builtins.print"):
                report = preview(cfg, options, recording=False)
                fallback.assert_not_called()
                constructor.assert_called_once()
                self.assertFalse(constructor.call_args.kwargs["headless"])
                self.assertIsNone(constructor.call_args.kwargs["enable_cameras"])
            self.assertEqual(report["steps"], 2)
            self.assertEqual(report["obs_shape"],
                             {key: [1, *shape] for key, shape in cfg.actor.model.observation_shapes.items()})
            self.assertTrue(report["environment_closed"])
            self.assertEqual(report["resets"], 2)
            native.reset.assert_called_once_with()
            native.close.assert_called_once()

    def test_lift_preview_reuses_the_same_native_environment_wrapper(self):
        from run import compose_project
        from shared.backends.isaac.environment import Environment
        from shared.integrations.rlinf.isaac import _ENVIRONMENT_FACTORIES
        from tools.preview import parse_options, preview

        cfg, _ = compose_project("assembly_lift")
        from tools.inspect_scene import apply_dimensions
        from shared.assembly.interfaces import load_interface
        report = load_interface(ROOT / "projects/assembly_lift/scene.yaml")
        if report is None:
            self.skipTest("run tools/inspect_scene.py assembly_lift first")
        apply_dimensions(cfg, report)
        options, _ = parse_options(["--steps", "2"], "view")
        observations = {key: torch.zeros(1, *shape)
                        for key, shape in cfg.actor.model.observation_shapes.items()}
        native = SimpleNamespace(
            num_envs=1, device="cpu", step_dt=0.02,
            action_manager=SimpleNamespace(total_action_dim=cfg.env.train.init_params.action_dim),
            sim=SimpleNamespace(is_stopped=Mock(return_value=False), is_playing=Mock(return_value=True)),
            reset=Mock(return_value=(observations, {})), close=Mock(),
            step=Mock(return_value=(observations, torch.ones(1), torch.zeros(1, dtype=torch.bool),
                                   torch.zeros(1, dtype=torch.bool), {"autoreset": True})),
        )
        environment = Environment(native)
        owner = SimpleNamespace(application=SimpleNamespace(is_running=Mock(return_value=True)))
        window_module = SimpleNamespace(get_default_app_window=Mock())
        with tempfile.TemporaryDirectory() as directory:
            cfg.runner.logger.log_path = directory
            constructor = Mock(return_value=(environment, owner))
            with patch.dict(_ENVIRONMENT_FACTORIES, {cfg.env.train.init_params.id: constructor}), \
                 patch("tools.preview.make_simulation") as fallback, \
                 patch.dict("os.environ", {"DISPLAY": ":test"}), \
                 patch.dict("sys.modules", {"omni.appwindow": window_module}), patch("builtins.print"):
                report = preview(cfg, options, recording=False)
                fallback.assert_not_called()
                constructor.assert_called_once()
                self.assertIsNone(constructor.call_args.kwargs["enable_cameras"])
            self.assertEqual(report["steps"], 2)
            self.assertEqual(report["obs_shape"],
                             {key: [1, *shape] for key, shape in cfg.actor.model.observation_shapes.items()})
            self.assertEqual(report["action_shape"], [1, cfg.env.train.init_params.action_dim])
            native.reset.assert_called_once_with()
            native.close.assert_called_once()

    def test_window_close_race_is_not_a_task_failure(self):
        from tools.preview import parse_options, preview

        options, _ = parse_options(["--steps", "4"], "view")
        for closing in (True, False):
            application = SimpleNamespace(is_running=Mock(return_value=True))

            def interrupted_step(actions):
                if closing:
                    application.is_running.return_value = False
                raise ReferenceError("physics object released")

            environment = SimpleNamespace(
                device="cpu", step_dt=1 / 60, action_manager=SimpleNamespace(total_action_dim=1),
                sim=SimpleNamespace(is_stopped=Mock(return_value=False), is_playing=Mock(return_value=True)),
                reset=Mock(return_value=({"policy": torch.zeros(1, 4)}, {})),
                step=Mock(side_effect=interrupted_step), close=Mock(),
            )
            window_module = SimpleNamespace(get_default_app_window=Mock())
            with tempfile.TemporaryDirectory() as directory:
                cfg = OmegaConf.create({"env": {"train": {"env_type": "isaaclab", "seed": 42,
                    "max_episode_steps": 8, "init_params": {"id": "test", "env_cfg": "test:Config",
                    "obs_dim": 4, "action_dim": 1}}}, "runner": {"logger": {"log_path": directory}}})
                with patch("tools.preview.make_simulation", return_value=(environment, SimpleNamespace(application=application))), \
                     patch.dict("os.environ", {"DISPLAY": ":test"}), \
                     patch.dict("sys.modules", {"omni.appwindow": window_module}), patch("builtins.print"):
                    if closing:
                        report = preview(cfg, options, recording=False)
                        self.assertEqual(report["status"], "stopped")
                        self.assertTrue(report["environment_closed"])
                    else:
                        with self.assertRaisesRegex(ReferenceError, "physics object"):
                            preview(cfg, options, recording=False)
                environment.close.assert_called_once()
