import os
import unittest
from unittest.mock import patch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pytest
import numpy as np

mujoco = pytest.importorskip("mujoco", reason="MuJoCo integration tests require the optional simulator", exc_type=ModuleNotFoundError)
pytest.importorskip("gym", reason="FrankaSim requires the optional Gym runtime", exc_type=ModuleNotFoundError)
pytest.importorskip("franka_sim", reason="FrankaSim is an optional simulator backend", exc_type=ModuleNotFoundError)

from projects.frankasim_stack_state.env import PandaStackStateGymEnv, STATE_DIM, STATE_FIELDS
from projects.frankasim_stack_v2.env import PandaStackProgressGymEnv


class StackStateTests(unittest.TestCase):
    def test_sample_budget_is_optional_and_uses_actual_replay_count(self):
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        import run
        from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner

        runner = SimpleNamespace(cfg=OmegaConf.create({"runner": {}}))
        self.assertFalse(AsyncEmbodiedRunner._sample_budget_reached(runner, {}))
        runner.cfg.runner.max_train_samples = 500000
        self.assertFalse(AsyncEmbodiedRunner._sample_budget_reached(runner, {"train/replay_buffer/total_samples": 499999}))
        self.assertTrue(AsyncEmbodiedRunner._sample_budget_reached(runner, {"train/replay_buffer/total_samples": 500224}))
        with self.assertRaisesRegex(ValueError, "cumulative replay"):
            AsyncEmbodiedRunner._sample_budget_reached(runner, {})

    def test_evaluation_seed_streams_are_reproducible_and_disjoint(self):
        from projects.frankasim_stack_state.check import evaluation_seeds

        first, second = evaluation_seeds(43, 32), evaluation_seeds(44, 32)
        self.assertEqual(first, evaluation_seeds(43, 32))
        self.assertEqual(len(set(first + second)), 64)

    def test_sample_gate_waits_for_new_data_and_spends_credit_once(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, Mock
        from omegaconf import OmegaConf
        import run
        from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncEmbodiedSACFSDPPolicy

        worker = SimpleNamespace(cfg=OmegaConf.create({"actor": {"min_new_samples_per_training_step": 64}}),
                                 replay_buffer=SimpleNamespace(total_samples=1000), should_stop=False)
        drains = []

        def drain(**kwargs):
            drains.append(kwargs)
            if len(drains) == 2:
                worker.replay_buffer.total_samples += 256

        worker._drain_received_trajectories = Mock(side_effect=drain)

        async def exercise():
            with patch("rlinf.workers.actor.async_fsdp_sac_policy_worker.asyncio.sleep", new_callable=AsyncMock) as wait:
                for index in range(4):
                    await AsyncEmbodiedSACFSDPPolicy._wait_for_new_samples(worker)
                    self.assertEqual(worker._sample_update_quota, 1000 + 64 * (index + 1))
                self.assertEqual(wait.await_count, 1)
            worker.cfg.actor.min_new_samples_per_training_step = 0
            await AsyncEmbodiedSACFSDPPolicy._wait_for_new_samples(worker)
            self.assertEqual(worker._sample_update_quota, 1256)

        asyncio.run(exercise())

    def test_native_sac_configuration_and_state_wrapper(self):
        from omegaconf import OmegaConf
        from run import compose_project
        from rlinf.envs import get_env_cls

        visual_config, _ = compose_project("frankasim_stack_v2")
        config, project = compose_project("frankasim_stack_state")
        project.preflight(config)
        self.assertFalse(config.actor.enable_drq)
        self.assertFalse(config.actor.fsdp_config.use_orig_params)
        self.assertIsNone(config.runner.resume_dir)
        self.assertIsNone(config.runner.ckpt_path)
        self.assertEqual(OmegaConf.to_container(config.algorithm, resolve=True),
                         OmegaConf.to_container(visual_config.algorithm, resolve=True))
        environment = get_env_cls(config.env.train.env_type, config.env.train)(config.env.train, 1, 43, 1, None)
        try:
            observation, _ = environment.reset(seed=43)
            self.assertEqual(set(observation), {"states"})
            self.assertEqual(tuple(observation["states"].shape), (1, STATE_DIM))
        finally:
            environment.close()
            for wrapped in environment.envs:
                wrapped.close()

    def test_supervisor_requires_both_independent_seeds_and_will_not_restart(self):
        import json
        from pathlib import Path
        import tempfile
        from projects.frankasim_stack_state.experiment import prepare, execute, write_json

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "state_run"
            prepare(output, epochs=2)

            def subprocess_fixture(command, **kwargs):
                if command[0] == "bash":
                    checkpoint = output / "runs/stack_state_sac/checkpoints/global_step_2/actor/model_state_dict/full_weights.pt"
                    checkpoint.parent.mkdir(parents=True)
                    checkpoint.write_bytes(b"mock checkpoint")
                else:
                    seed = int(command[command.index("--seed") + 1])
                    destination = Path(command[command.index("--output") + 1])
                    destination.mkdir()
                    count = 32 if seed == 43 else 24
                    from projects.frankasim_stack_state.check import evaluation_seeds

                    write_json(destination / "report.json", {
                        "seed": seed, "episodes": 32, "policy_input": "privileged_state",
                        "reset_seeds": evaluation_seeds(seed, 32),
                        "success_rate": count / 32,
                        "records": [{"success": index < count} for index in range(32)],
                    })

            with patch("projects.frankasim_stack_state.experiment.subprocess.run", side_effect=subprocess_fixture) as process:
                execute(output)
                self.assertEqual(process.call_count, 3)
                with self.assertRaisesRegex(ValueError, "Refusing to restart"):
                    execute(output)
                self.assertEqual(process.call_count, 3)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["state"], "acceptance_failed")
            self.assertEqual(status["success_rate"], 0.875)
            self.assertEqual(status["episodes"], 64)
            self.assertFalse(status["accepted"])

    def test_native_mlp_bounded_actions_and_ten_q_gradients(self):
        import torch
        from run import compose_project
        from rlinf.models import get_model
        from rlinf.models.embodiment.base_policy import ForwardType

        config, _ = compose_project("frankasim_stack_state")
        torch.manual_seed(42)
        model = get_model(config.actor.model)
        observation = {"states": torch.randn(8, STATE_DIM)}
        actions, result = model.predict_action_batch(observation, mode="eval")
        self.assertEqual(tuple(actions.shape), (8, 1, 4))
        self.assertTrue((actions.abs() <= 1).all())
        self.assertEqual(set(result["forward_inputs"]), {"states", "action", "model_action"})
        sampled, logprobs, _ = model(forward_type=ForwardType.SAC, obs=observation)
        values = model(forward_type=ForwardType.SAC_Q, obs=observation, actions=sampled)
        self.assertEqual(model.q_head.num_q_heads, 10)
        self.assertEqual(values.shape[0], 8)
        self.assertEqual(values.numel(), 8 * 10)
        loss = values.square().mean() + logprobs.square().mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_observation_has_explicit_state_only_layout(self):
        environment = PandaStackStateGymEnv()
        try:
            with patch.object(environment, "render", side_effect=AssertionError("Training must not render")):
                observation, _ = environment.reset(seed=43)
                self.assertTrue(environment.observation_space.contains(observation))
                self.assertEqual(set(observation), {"state"})
                state = observation["state"]["stack_state"]
                self.assertEqual(state.shape, (STATE_DIM,))
                self.assertEqual(sum(size for _, size in STATE_FIELDS), STATE_DIM)
                self.assertTrue(np.isfinite(state).all())
                np.testing.assert_array_equal(state[24:31], environment.data.jnt("block").qpos.astype(np.float32))
                np.testing.assert_array_equal(state[-3:], (environment.data.jnt("support").qpos[:3]
                                              - environment.data.jnt("block").qpos[:3]).astype(np.float32))
                environment.step(np.zeros(4))
        finally:
            environment.close()

    def test_physics_rewards_and_success_match_visual_v2(self):
        visual = PandaStackProgressGymEnv()
        state = PandaStackStateGymEnv()
        frames = {side: np.zeros((1, 1, 3), dtype=np.uint8) for side in ("front", "wrist")}
        try:
            with patch.object(visual, "render", return_value=frames):
                for seed in (43, 44):
                    visual.reset(seed=seed)
                    state.reset(seed=seed)
                    generator = np.random.RandomState(seed)
                    for action in generator.uniform(-0.1, 0.1, size=(12, 4)):
                        expected = visual.step(action)
                        actual = state.step(action)
                        np.testing.assert_array_equal(state.data.qpos, visual.data.qpos)
                        np.testing.assert_array_equal(state.data.qvel, visual.data.qvel)
                        self.assertEqual(actual[1:], expected[1:])
        finally:
            visual.close()
            state.close()

    def test_actual_support_and_release_hold_are_still_required(self):
        environment = PandaStackStateGymEnv()
        try:
            environment.reset(seed=43)
            environment.data.jnt("block").qpos[:3] = environment.data.jnt("support").qpos[:3] + [0, 0, 0.04]
            mujoco.mj_forward(environment.model, environment.data)
            for step in range(30):
                _, reward, _, _, info = environment.step(np.zeros(4))
                if step < 24:
                    self.assertFalse(info["success"])
            self.assertTrue(info["success"])
            self.assertEqual(reward, 5.0)
            self.assertGreater(info["stack_metrics"]["support_force"], 0.05)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
