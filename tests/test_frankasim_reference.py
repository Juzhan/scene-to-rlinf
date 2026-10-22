import unittest

from tools.frankasim_reference import accepted, make_config, stable_grasp


class FrankaSimReferenceTests(unittest.TestCase):
    def test_official_recipe_retains_algorithm_model_and_actions(self):
        config, _ = make_config("/tmp/frankasim_reference_test", 2)
        self.assertEqual(config.algorithm.loss_type, "embodied_sac")
        self.assertTrue(config.actor.enable_drq)
        self.assertEqual(config.actor.model.action_dim, 4)
        self.assertEqual(config.actor.model.state_dim, 7)
        self.assertEqual(config.actor.model.image_num, 2)
        self.assertEqual(list(config.env.train.render_size), [720, 1280])
        self.assertEqual(list(config.env.train.resize_to), [128, 128])
        self.assertEqual(config.actor.global_batch_size, 256)
        self.assertEqual(config.algorithm.replay_buffer.sample_window_size, 256)
        self.assertEqual(config.runner.save_interval % config.runner.val_check_interval, 0)

    def test_workspace_recipe_matches_reference_training_contract(self):
        from importlib.util import find_spec
        from omegaconf import OmegaConf
        from projects.frankasim import project

        reference, _ = make_config("/tmp/frankasim_workspace_test", 2)
        workspace, _ = make_config("/tmp/frankasim_workspace_test", 2, workspace=True)
        for section in ("algorithm", "cluster", "actor", "rollout", "env", "weight_syncer", "reward", "critic"):
            with self.subTest(section=section):
                self.assertEqual(OmegaConf.to_container(workspace[section], resolve=True),
                                 OmegaConf.to_container(reference[section], resolve=True))
        self.assertTrue(workspace.runner.async_training)
        self.assertIsNone(workspace.runner.resume_dir)
        self.assertIsNone(workspace.runner.ckpt_path)
        missing = [name for name in ("gym", "mujoco", "franka_sim") if find_spec(name) is None]
        if missing:
            with self.assertRaises(RuntimeError) as error:
                project.preflight(workspace)
            self.assertEqual(str(error.exception),
                             "FrankaSim runtime is not ready; use .venv-frankasim: " + "; ".join(missing))
        else:
            project.preflight(workspace)

    def test_workspace_driver_uses_project_entry_and_same_evaluation(self):
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import patch
        from tools.frankasim_reference import ROOT, execute, prepare, write_json

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiment"
            plan = prepare(output, 2, workspace=True)
            self.assertIn("run.py", plan["source_sha256"])
            checkpoint = output / "runs/frankasim/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            commands = []

            def complete(command, **kwargs):
                commands.append(command)
                if "evaluate" in command:
                    seed = int(command[command.index("--seed") + 1])
                    destination = command[command.index("--output") + 1]
                    write_json(destination, {"seed": seed, "episodes": 32, "success_rate": 0.9,
                                             "held_0_5s_rate": 0.9, "frozen_encoder_verified": True})

            with patch("tools.frankasim_reference.subprocess.run", side_effect=complete):
                execute(output)
            self.assertEqual(commands[0][:4], [sys.executable, str(ROOT / "run.py"), "train", "frankasim"])
            self.assertEqual(len(commands), 3)
            self.assertTrue(all("--workspace" in command for command in commands[1:]))
            import json
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["phase"], "completed")
            self.assertTrue(status["official_accepted"])

    def test_ppo_resume_freezes_checkpoint_and_total_budget(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        from tools.frankasim_reference import execute, prepare

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "global_step_1062"
            for name in ("model_state_dict/full_weights.pt", "dcp_checkpoint/.metadata",
                         "dcp_checkpoint/__0_0.distcp"):
                artifact = checkpoint / "actor" / name
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.touch()
            output = Path(directory) / "experiment"
            plan = prepare(output, 15000, workspace=True, recipe="ppo", resume_dir=checkpoint)
            self.assertEqual(plan["config"]["runner"]["resume_dir"], str(checkpoint))
            self.assertEqual(plan["resume"]["global_step"], 1062)
            self.assertEqual(plan["resume"]["additional_epochs"], 13938)
            self.assertEqual(plan["resume"]["additional_transitions"], 3568128)
            self.assertEqual(plan["ablation"]["planned_transitions"], 3840000)
            self.assertEqual(len(plan["resume"]["checkpoint_sha256"]), 3)
            (checkpoint / "actor/dcp_checkpoint/__0_0.distcp").unlink()
            with patch("tools.frankasim_reference.subprocess.run") as launch:
                with self.assertRaises(FileNotFoundError):
                    execute(output)
                launch.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Total epochs"):
                prepare(Path(directory) / "invalid", 1062, workspace=True, recipe="ppo", resume_dir=checkpoint)

    def test_ppo_bounded_action_logprobs_and_gradients(self):
        import torch
        from projects.frankasim.policy import build_model

        config, _ = make_config("/tmp/frankasim_ppo_probability_test", 2, workspace=True, recipe="ppo")
        torch.manual_seed(42)
        model = build_model(config.actor.model, torch.float32).eval()
        observations = {"states": torch.zeros(2, 7),
                        "main_images": torch.randint(0, 256, (2, 128, 128, 3), dtype=torch.uint8),
                        "extra_view_images": torch.randint(0, 256, (2, 1, 128, 128, 3), dtype=torch.uint8)}
        with torch.no_grad():
            model.actor_logstd.fill_(1.5)
            actions, collected = model.predict_action_batch(observations)
        inputs = {name: value.clone() for name, value in collected["forward_inputs"].items()}
        model.train()
        recomputed = model.default_forward(inputs)
        torch.testing.assert_close(collected["prev_logprobs"], recomputed["logprobs"])
        torch.testing.assert_close(collected["prev_values"], recomputed["values"])
        torch.testing.assert_close(recomputed["gaussian_kl"], torch.zeros(2))
        self.assertTrue((actions.abs() <= 1).all())
        self.assertTrue((inputs["action"].abs() > 1).any())
        (recomputed["logprobs"].mean() + recomputed["values"].mean()).backward()
        self.assertIsNotNone(model.actor_mean.weight.grad)
        self.assertTrue(torch.isfinite(model.actor_mean.weight.grad).all())
        self.assertTrue(all(parameter.grad is None for encoder in model.encoders
                            for parameter in encoder.resnet_backbone.parameters()))

    def test_ppo_recipe_preserves_task_and_actor_contract(self):
        from omegaconf import OmegaConf
        from run import compose_project

        sac, _ = compose_project("frankasim")
        ppo, _ = compose_project("frankasim", "ppo")
        self.assertEqual(OmegaConf.to_container(sac.env, resolve=True),
                         OmegaConf.to_container(ppo.env, resolve=True))
        for key in ("state_dim", "action_dim", "image_num", "model_path", "image_size", "num_action_chunks"):
            self.assertEqual(sac.actor.model[key], ppo.actor.model[key])
        self.assertEqual(sac.algorithm.gamma, ppo.algorithm.gamma)
        self.assertEqual(sac.actor.optim.lr, ppo.actor.optim.lr)
        self.assertFalse(ppo.runner.async_training)
        self.assertEqual(ppo.runner.weight_sync_interval, 1)
        self.assertTrue(ppo.actor.model.independent_std)
        self.assertEqual(ppo.algorithm.adaptive_kl_target, 0.01)
        self.assertFalse(ppo.actor.enable_drq)
        self.assertTrue(ppo.actor.model.add_value_head)
        self.assertFalse(ppo.actor.model.add_q_head)
        self.assertTrue(ppo.rollout.collect_prev_infos)
        self.assertFalse(ppo.rollout.collect_transitions)
        self.assertEqual(ppo.runner.max_epochs * ppo.env.train.max_steps_per_rollout_epoch, 271872)

    def test_ppo_promotion_requires_comparable_success_and_holding(self):
        from tools.frankasim_reference import ablation_comparison

        baseline = [{"seed": seed, "episodes": 32, "success_rate": success,
                     "held_0_5s_rate": holding, "frozen_encoder_verified": True}
                    for seed, success, holding in ((43, 1., 12/32), (44, 31/32, 19/32))]
        candidate = [dict(report, success_rate=30/32) for report in baseline]
        self.assertTrue(ablation_comparison(candidate, baseline)["passed"])
        candidate[0]["held_0_5s_rate"] = 11/32
        self.assertFalse(ablation_comparison(candidate, baseline)["passed"])
        candidate[0]["held_0_5s_rate"] = 12/32
        candidate[0]["success_rate"] = 29/32
        self.assertFalse(ablation_comparison(candidate, baseline)["passed"])

    def test_grasp_requires_height_speed_proximity_and_both_contacts(self):
        self.assertTrue(stable_grasp(0.21, 0.01, 0.03, 1., 1.))
        for values in ((0.1, 0.01, 0.03, 1., 1.), (0.21, 1., 0.03, 1., 1.),
                       (0.21, 0.01, 0.2, 1., 1.), (0.21, 0.01, 0.03, 0., 1.),
                       (0.21, 0.01, 0.03, 1., 0.)):
            self.assertFalse(stable_grasp(*values))

    def test_acceptance_requires_independent_seeds_and_physical_holding(self):
        reports = [{"seed": seed, "episodes": 32, "success_rate": 1.,
                    "held_0_5s_rate": 1., "frozen_encoder_verified": True} for seed in (43, 44)]
        self.assertTrue(accepted(reports))
        reports[1]["held_0_5s_rate"] = 0.
        self.assertFalse(accepted(reports))
        reports[1]["held_0_5s_rate"] = 1.
        reports[1]["frozen_encoder_verified"] = False
        self.assertFalse(accepted(reports))


if __name__ == "__main__":
    unittest.main()
