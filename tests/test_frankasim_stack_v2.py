import unittest

import pytest
import numpy as np
import torch

mujoco = pytest.importorskip("mujoco", reason="MuJoCo integration tests require the optional simulator", exc_type=ModuleNotFoundError)
pytest.importorskip("gym", reason="FrankaSim requires the optional Gym runtime", exc_type=ModuleNotFoundError)
pytest.importorskip("franka_sim", reason="FrankaSim is an optional simulator backend", exc_type=ModuleNotFoundError)

from projects.frankasim_stack_v2.env import PandaStackProgressGymEnv, staged_reward


class StackProgressTests(unittest.TestCase):
    def test_actor_is_inherited_but_critic_is_fresh(self):
        from run import compose_project
        from projects.frankasim.policy import build_model as build_native
        from projects.frankasim_stack_v2.policy import build_model

        config, project = compose_project("frankasim_stack_v2")
        project.preflight(config)
        previous = torch.load(config.actor.model.initial_actor_checkpoint, map_location="cpu", weights_only=True)
        torch.manual_seed(42)
        initial = build_native(config.actor.model, torch.float32).state_dict()
        torch.manual_seed(42)
        warmed = build_model(config.actor.model, torch.float32).state_dict()
        for name, value in warmed.items():
            torch.testing.assert_close(value, initial[name] if name.startswith("q_head.") else previous[name], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(warmed[name], previous[name]) for name in warmed if name.startswith("q_head.")))

    def test_transport_has_signal_over_original_random_distances(self):
        rewards = [staged_reward(0.005, 0.08, distance, 0.04, False, False)
                   for distance in (0.4, 0.3, 0.2, 0.1)]
        self.assertTrue(all(after > before + 0.05 for before, after in zip(rewards, rewards[1:])))
        self.assertGreater(rewards[1], 0.3)

    def test_lift_plateau_is_not_increased_by_hovering_higher_or_tcp_closeness(self):
        base = staged_reward(0.005, 0.08, 0.3, 0.04, False, False)
        higher = staged_reward(0.005, 0.18, 0.3, 0.14, False, False)
        away = staged_reward(0.1, 0.08, 0.3, 0.04, False, False)
        self.assertLessEqual(higher, base)
        self.assertAlmostEqual(base, away)

    def test_placement_contact_release_and_success_outrank_hovering(self):
        hover = staged_reward(0.005, 0.08, 0.0, 0.04, False, False)
        lowered = staged_reward(0.005, 0.04, 0.0, 0.0, False, False)
        contact = staged_reward(0.005, 0.04, 0.0, 0.0, True, False)
        released = staged_reward(0.1, 0.04, 0.0, 0.0, True, True)
        self.assertLess(hover, lowered)
        self.assertLess(lowered, contact)
        self.assertLess(contact, released)

    def test_original_actual_contact_and_hold_criterion_are_unchanged(self):
        environment = PandaStackProgressGymEnv()
        try:
            observation, _ = environment.reset(seed=43)
            self.assertEqual(set(observation["state"]), {"panda/tcp_pos", "panda/tcp_vel", "panda/gripper_pos"})
            bottom = environment.data.jnt("support").qpos[:3].copy()
            environment.data.jnt("block").qpos[:3] = bottom + [0.0, 0.0, 0.04]
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
