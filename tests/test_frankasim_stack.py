import unittest

import pytest
import numpy as np

mujoco = pytest.importorskip("mujoco", reason="MuJoCo integration tests require the optional simulator", exc_type=ModuleNotFoundError)
pytest.importorskip("gym", reason="FrankaSim requires the optional Gym runtime", exc_type=ModuleNotFoundError)
pytest.importorskip("franka_sim", reason="FrankaSim is an optional simulator backend", exc_type=ModuleNotFoundError)

from projects.frankasim_stack.env import PandaStackCubeGymEnv, shaped_reward


class StackTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = PandaStackCubeGymEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_images_robot_only_and_reproducible_sampling(self):
        observation, _ = self.env.reset(seed=43)
        np.testing.assert_array_equal(self.env._panda_dof_ids,
                                      self.env.model.jnt_dofadr[self.env._panda_dof_ids])
        self.assertEqual(set(observation["state"]), {"panda/tcp_pos", "panda/tcp_vel", "panda/gripper_pos"})
        for frame in observation["images"].values():
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertGreater(float(frame.std()), 1.0)
        positions = self.env.data.qpos.copy()
        self.env.reset(seed=43)
        np.testing.assert_array_equal(self.env.data.qpos, positions)
        self.assertFalse(self.env.stack_metrics()["placed"])

    def test_native_sac_config_and_wrapper_keep_robot_only_two_views(self):
        from run import compose_project
        from omegaconf import OmegaConf
        from rlinf.envs import get_env_cls

        config, project = compose_project("frankasim_stack")
        reference, _ = compose_project("frankasim")
        project.preflight(config)
        for section in ("algorithm", "actor", "rollout", "weight_syncer"):
            self.assertEqual(OmegaConf.to_container(config[section], resolve=True),
                             OmegaConf.to_container(reference[section], resolve=True))
        wrapper = get_env_cls(config.env.train.env_type, config.env.train)(config.env.train, 1, 43, 1, None)
        try:
            observation, _ = wrapper.reset()
            self.assertEqual(set(observation), {"states", "main_images", "extra_view_images", "task_descriptions"})
            self.assertEqual(tuple(observation["states"].shape), (1, 7))
            self.assertEqual(tuple(observation["main_images"].shape), (1, 128, 128, 3))
            self.assertEqual(tuple(observation["extra_view_images"].shape), (1, 1, 128, 128, 3))
        finally:
            wrapper.close()
            for environment in wrapper.envs:
                environment.close()

    def test_floating_cube_and_closed_gripper_are_not_success(self):
        self.env.reset(seed=43)
        bottom = self.env.data.jnt("support").qpos[:3].copy()
        self.env.data.jnt("block").qpos[:3] = bottom + np.array([0.0, 0.0, 0.04])
        mujoco.mj_forward(self.env.model, self.env.data)
        self.assertFalse(self.env.stack_metrics()["success"])
        self.env.data.ctrl[self.env._gripper_ctrl_id] = 255
        self.assertFalse(self.env.stack_metrics()["released"])

    def test_released_stack_reward_beats_grasp_or_lift(self):
        lifted = shaped_reward(0.0, 0.08, 0.2, 0.04, False, False)
        stacked = shaped_reward(0.1, 0.04, 0.0, 0.0, True, True)
        self.assertGreater(stacked, lifted)

    def test_actual_support_contact_requires_continuous_released_hold(self):
        self.env.reset(seed=43)
        bottom = self.env.data.jnt("support").qpos[:3].copy()
        self.env.data.jnt("block").qpos[:3] = bottom + np.array([0.0, 0.0, 0.04])
        mujoco.mj_forward(self.env.model, self.env.data)
        for step in range(30):
            _, _, _, _, info = self.env.step(np.zeros(4))
            if step < 24:
                self.assertFalse(info["success"])
        self.assertTrue(info["success"])
        self.assertGreater(info["stack_metrics"]["support_force"], 0.05)
        self.env.data.jnt("block").qpos[0] += 0.1
        mujoco.mj_forward(self.env.model, self.env.data)
        _, _, _, _, info = self.env.step(np.zeros(4))
        self.assertFalse(info["success"])
        self.assertEqual(info["stack_metrics"]["hold_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
