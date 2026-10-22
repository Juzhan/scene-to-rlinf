import unittest
import math
from pathlib import Path
import xml.etree.ElementTree as ElementTree

import torch

from projects.franka_isaac.contract import actor_observation, integrate_gripper, integrate_target, lift_reward
from projects.franka_isaac.contract import (FRONT_CAMERA_POSITION, FRONT_CAMERA_ORIENTATION, FRONT_CAMERA_VERTICAL_FOV,
                                          WRIST_CAMERA_POSITION, WRIST_CAMERA_ORIENTATION, WRIST_CAMERA_VERTICAL_FOV,
                                          camera_lens_parameters)


class IsaacVisualLiftTests(unittest.TestCase):
    def test_camera_geometry_matches_reference_and_panda_mount(self):
        source = Path(__file__).resolve().parents[1] / "third_party/serl-frankasim/franka_sim/franka_sim/envs/xmls"
        front = ElementTree.parse(source / "arena.xml").find("./worldbody/camera[@name='front']")
        wrist = ElementTree.parse(source / "panda.xml").find(".//camera[@name='handcam_depth']")
        self.assertEqual(FRONT_CAMERA_POSITION, tuple(map(float, front.get("pos").split())))
        quaternion = torch.tensor(list(map(float, front.get("quat").split())), dtype=torch.float64)
        torch.testing.assert_close(torch.tensor(FRONT_CAMERA_ORIENTATION, dtype=torch.float64),
                                   quaternion / torch.linalg.vector_norm(quaternion))
        self.assertEqual(FRONT_CAMERA_VERTICAL_FOV, float(front.get("fovy")))
        position = tuple(map(float, wrist.get("pos").split()))
        self.assertEqual(WRIST_CAMERA_POSITION, (-position[0], -position[1], position[2]))
        quaternion = list(map(float, wrist.get("quat").split()))
        norm = math.sqrt(sum(value * value for value in quaternion))
        torch.testing.assert_close(torch.tensor(WRIST_CAMERA_ORIENTATION, dtype=torch.float64),
                       torch.tensor((-quaternion[3] / norm, -quaternion[2] / norm,
                             quaternion[1] / norm, quaternion[0] / norm), dtype=torch.float64))
        self.assertEqual(WRIST_CAMERA_VERTICAL_FOV, float(wrist.get("fovy")))
        for fov in (FRONT_CAMERA_VERTICAL_FOV, WRIST_CAMERA_VERTICAL_FOV):
            lens = camera_lens_parameters(fov)
            actual = math.degrees(2.0 * math.atan(lens["vertical_aperture"] / (2.0 * lens["focal_length"])))
            self.assertAlmostEqual(actual, fov)
            self.assertAlmostEqual(lens["horizontal_aperture"] / lens["vertical_aperture"], 1280 / 720)

    def test_target_accumulates_actions_and_clips_original_workspace(self):
        target = torch.tensor([[0.4, 0.0, 0.3]])
        action = torch.tensor([[1.0, -1.0, 1.0]])
        target = integrate_target(target, action)
        torch.testing.assert_close(target, torch.tensor([[0.5, -0.1, 0.4]]))
        target = integrate_target(integrate_target(target, action), action)
        torch.testing.assert_close(target, torch.tensor([[0.6, -0.3, 0.5]]))

    def test_reward_matches_original_and_gripper_is_incremental(self):
        position = torch.zeros(2, 3)
        objects = position.clone()
        objects[1, 2] = 0.2
        expected = torch.tensor([0.3, 0.7 + 0.3 * torch.exp(torch.tensor(-4.0))])
        torch.testing.assert_close(lift_reward(position, objects, torch.zeros(2)), expected)
        torch.testing.assert_close(integrate_gripper(torch.tensor([0.2, 0.9]), torch.tensor([-0.4, 0.3])),
                                   torch.tensor([0.0, 1.0]))

    def test_actor_state_is_robot_only_and_two_real_720p_views(self):
        position = torch.zeros(1, 3)
        rgb = torch.zeros(1, 720, 1280, 4, dtype=torch.uint8)
        observation = actor_observation(position, position, torch.zeros(1), rgb, rgb)
        self.assertEqual(set(observation), {"states", "main_images", "extra_view_images"})
        self.assertEqual(observation["states"].shape, (1, 7))
        self.assertEqual(observation["main_images"].shape, (1, 128, 128, 3))
        self.assertEqual(observation["extra_view_images"].shape, (1, 1, 128, 128, 3))
        with self.assertRaisesRegex(ValueError, "Both cameras"):
            actor_observation(position, position, torch.zeros(1), rgb[:, :128, :128], rgb)

    def test_sac_learning_configuration_is_unchanged(self):
        from run import compose_project
        from omegaconf import OmegaConf

        isaac, _ = compose_project("franka_isaac")
        reference, _ = compose_project("frankasim")
        for section in ("algorithm", "actor", "rollout", "weight_syncer"):
            self.assertEqual(OmegaConf.to_container(isaac[section], resolve=True),
                             OmegaConf.to_container(reference[section], resolve=True))
        self.assertEqual(isaac.env.train.env_type, "isaaclab")
        self.assertEqual(isaac.env.train.max_episode_steps, 100)
        self.assertTrue(isaac.runner.async_training)


if __name__ == "__main__":
    unittest.main()