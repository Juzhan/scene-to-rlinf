"""Select one task while retaining LeRobot's official LIBERO evaluation loop."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


REVISION = "31d453f7edd78c839a8bbc39744a292686daf0de"
MODEL_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"


def sample_bowl_plate_positions(positions, robot_base, seed, *, centered=False):
    import numpy as np

    generator = np.random.default_rng(seed)
    center = np.asarray(robot_base, dtype=float)[:2]
    for attempt in range(100):
        sampled = {}
        target_middle = center[1] + generator.uniform(-0.025, 0.025) if centered else None
        reference_middle = np.mean([position[1] for position in positions.values()])
        for name in sorted(positions):
            position = np.asarray(positions[name], dtype=float).copy()
            if centered:
                position[1] += target_middle - reference_middle + generator.uniform(-0.005, 0.005)
            else:
                angle = generator.uniform(-0.10, 0.10)
                rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                position[:2] = center + rotation @ (position[:2] - center)
            sampled[name] = position
        separation = np.linalg.norm(sampled["akita_black_bowl_1"][:2] - sampled["plate_1"][:2])
        displacements = [np.linalg.norm(sampled[name][:2] - np.asarray(positions[name])[:2]) for name in sampled]
        lateral_inward = all(abs(sampled[name][1] - center[1]) < abs(positions[name][1] - center[1]) for name in sampled)
        valid_shift = lateral_inward if centered else all(0.015 <= distance <= 0.06 for distance in displacements)
        if 0.11 <= separation <= 0.20 and valid_shift:
            return sampled
    raise ValueError("Could not sample separated bowl/plate positions near the reference robot distance")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/smolvla_libero/model"))
    parser.add_argument("--output", type=Path, default=Path("outputs/smolvla_libero/native_task2_20261010"))
    parser.add_argument("--task-id", type=int, default=2)
    parser.add_argument("--suite", choices=("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--grasp-black-bowl", action="store_true",
                        help="Keep the original scene and evaluate grasping any black bowl instead of the benchmark goal")
    parser.add_argument("--stack-two-bowls", action="store_true",
                        help="Remove plate_1 and black bowl 1, then stack either remaining black bowl on the other")
    parser.add_argument("--bowl-on-plate", action="store_true",
                        help="Keep only the original target black bowl and plate, with the original instruction and goal")
    parser.add_argument("--clear-table", action="store_true")
    parser.add_argument("--closer-m", type=float, default=0.0)
    parser.add_argument("--layout-seed", type=int)
    parser.add_argument("--vary-layout", action="store_true")
    parser.add_argument("--center-layout", action="store_true")
    parser.add_argument("--export-scene", action="store_true")
    parser.add_argument("--residual-steps", type=int, default=0)
    parser.add_argument("--rlinf-ppo-steps", type=int, default=0)
    parser.add_argument("--residual-start-step", type=int)
    parser.add_argument("--teacher-corrections", action="store_true")
    parser.add_argument("--takeover-step", type=int, default=180)
    parser.add_argument("--teacher-target", type=int, default=0)
    parser.add_argument("--placement-only", action="store_true")
    parser.add_argument("--supervised-data", type=Path)
    parser.add_argument("--supervised-epochs", type=int, default=500)
    parser.add_argument("--learner-checkpoint", type=Path)
    parser.add_argument("--resume-residual", type=Path)
    parser.add_argument("--additional-corrections", type=Path, action="append", default=[])
    parser.add_argument("--evaluate-residual", type=Path)
    parser.add_argument("--compare-residual", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--compare-baseline", action="store_true")
    parser.add_argument("--evaluate-pi05", action="store_true")
    args = parser.parse_args()
    if args.rlinf_ppo_steps:
        valid_scene = (args.resume_residual and args.bowl_on_plate and args.clear_table and args.center_layout
                       and args.vary_layout and args.suite == "libero_spatial" and args.task_id == 8
                       and args.closer_m == .10 and args.residual_start_step == 10)
        conflicting = any((args.evaluate_residual, args.supervised_data, args.teacher_corrections,
                           args.residual_steps, args.compare_residual, args.evaluate_pi05,
                           args.export_scene, args.additional_corrections))
        if args.rlinf_ppo_steps < 256 or args.rlinf_ppo_steps % 256 or not valid_scene or conflicting:
            parser.error("RLinf PPO requires centered varied spatial task 8, closer-m .10, gate 10, resume-residual and a 256-step multiple")
    if args.vary_layout and (args.layout_seed is None or not args.center_layout):
        parser.error("vary-layout requires layout-seed and center-layout")
    if args.residual_start_step is not None and not (0 <= args.residual_start_step < 280
                                                   and (args.teacher_corrections or args.evaluate_residual or args.rlinf_ppo_steps)):
        parser.error("residual-start-step requires correction collection or residual evaluation and a value in 0..279")
    if args.center_layout and args.layout_seed is None:
        parser.error("center-layout requires layout-seed")
    if args.layout_seed is not None and not (args.bowl_on_plate and args.clear_table and (args.episodes == 1 or args.vary_layout)
                                             and (args.evaluate_pi05 or args.evaluate_residual or args.teacher_corrections
                                                  or args.rlinf_ppo_steps or args.export_scene)):
        parser.error("layout-seed requires bowl/plate evaluation, correction collection or scene export; multiple episodes require vary-layout")
    if args.evaluate_pi05:
        if not (args.bowl_on_plate and args.clear_table and args.suite == "libero_spatial" and args.task_id == 8):
            parser.error("PI0.5 evaluation requires spatial task 8, bowl/plate and clear table")
        if args.evaluate_residual or args.compare_residual or args.supervised_data or args.teacher_corrections or args.residual_steps or args.export_scene:
            parser.error("PI0.5 evaluation cannot be combined with other evaluation, training or collection modes")
    if args.compare_baseline and (not args.evaluate_residual or args.compare_residual or args.baseline_only):
        parser.error("compare-baseline requires evaluate-residual without compare-residual or baseline-only")
    if args.baseline_only and not args.evaluate_residual:
        parser.error("baseline-only requires the residual evaluation checkpoint pair")
    if args.evaluate_residual or args.compare_residual:
        if not (args.evaluate_residual and (args.compare_residual or args.compare_baseline) and args.bowl_on_plate and args.clear_table
                and args.suite == "libero_spatial" and args.task_id == 8):
            parser.error("Residual evaluation requires a checkpoint comparison and spatial task 8, bowl/plate, clear table")
        if args.supervised_data or args.teacher_corrections or args.residual_steps or args.export_scene:
            parser.error("Residual evaluation cannot be combined with training, collection or scene export")
    if (args.resume_residual or args.additional_corrections) and not args.rlinf_ppo_steps and not (args.supervised_data and args.resume_residual and args.additional_corrections):
        parser.error("Continuation requires supervised-data, resume-residual and additional-corrections together")
    if args.learner_checkpoint and not (args.teacher_corrections and args.placement_only and args.closer_m == 0.05):
        parser.error("learner-checkpoint requires teacher-corrections, placement-only, and closer-m 0.05")
    if args.supervised_data and (args.teacher_corrections or args.residual_steps or not (args.bowl_on_plate and args.clear_table) or args.closer_m != 0.05):
        parser.error("Supervised bowl training requires only bowl/plate, clear table, closer-m 0.05 and no other training mode")
    if args.supervised_epochs < 1:
        parser.error("supervised-epochs must be positive")
    if args.teacher_target < 0 or (args.teacher_target and not args.teacher_corrections):
        parser.error("teacher-target requires teacher-corrections and must be nonnegative")
    if args.placement_only and not (args.teacher_corrections or args.supervised_data or args.evaluate_residual or args.evaluate_pi05):
        parser.error("placement-only requires teacher-corrections, supervised-data or a policy evaluation mode")
    if args.teacher_corrections and (args.residual_steps or not (args.bowl_on_plate and args.clear_table)):
        parser.error("Teacher correction requires --bowl-on-plate --clear-table without residual training")
    if not 0 <= args.takeover_step < 280:
        parser.error("takeover-step must be between 0 and 279")
    if args.residual_steps < 0 or (args.residual_steps and not (args.bowl_on_plate and args.clear_table)):
        parser.error("Residual training requires --bowl-on-plate --clear-table and nonnegative steps")
    if (args.clear_table or args.closer_m) and not args.bowl_on_plate:
        parser.error("Table clearing and translation require --bowl-on-plate")
    if not 0 <= args.closer_m <= 0.2:
        parser.error("closer-m must be between 0 and 0.2")
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if sum((args.grasp_black_bowl, args.stack_two_bowls, args.bowl_on_plate)) > 1:
        parser.error("Choose only one task variant")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import gymnasium as gym
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.envs.libero import create_libero_envs
    from lerobot.envs.utils import close_envs
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.random_utils import set_seed

    with (args.checkpoint / "model.safetensors").open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
            raise ValueError("Checkpoint is not the pinned official SmolVLA LIBERO model")
    saved = json.loads((args.checkpoint / "train_config.json").read_text())
    settings = saved["env"]
    fields = ("task", "fps", "episode_length", "obs_type", "render_mode", "camera_name",
              "init_states", "camera_name_mapping", "observation_height", "observation_width", "control_mode")
    environment_config = LiberoEnv(**{key: settings[key] for key in fields})
    if args.suite is not None:
        environment_config.task = args.suite
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.pretrained_path = args.checkpoint
    config.device = "cpu" if args.check else "cuda"
    config.load_vlm_weights = False
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": config.device},
                                "rename_observations_processor": {"rename_map": saved["rename_map"]}},
    )
    statistics = {}
    for feature, pipeline in (("observation.state", preprocessor), ("action", postprocessor)):
        matches = [step.stats[feature] for step in pipeline.steps
                   if feature in getattr(step, "stats", {})]
        if len(matches) != 1:
            raise ValueError(f"Missing official processor statistics: {feature}")
        mean, std = (torch.as_tensor(matches[0][key]) for key in ("mean", "std"))
        if mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std < 0).any():
            raise ValueError(f"Invalid official processor statistics: {feature}")
        statistics[feature] = list(mean.shape)
    provenance = {"model": "lerobot/smolvla_libero", "revision": REVISION,
                  "sha256": MODEL_SHA256, "suite": environment_config.task,
                  "task_id": args.task_id, "seed": args.seed, "episodes": args.episodes,
                  "statistics_shapes": statistics, "rename_map": saved["rename_map"],
                  "trained": False, "residual": False}
    if args.grasp_black_bowl:
        provenance.update(instruction_override="pick up the black bowl.", official_benchmark_score=False,
                          success_criterion="same black bowl lifted >= 0.05m with bilateral finger contact for 0.5s")
    if args.stack_two_bowls:
        provenance.update(instruction_override="pick up a black bowl and place it on the other black bowl.",
                          removed_objects=["plate_1", "akita_black_bowl_1"], official_benchmark_score=False,
                          success_criterion="either native On direction, bottom on table, top off table, released and stationary for 0.5s")
    if args.bowl_on_plate:
        provenance.update(retained_objects=["akita_black_bowl_1", "plate_1"], original_instruction_and_goal=True,
                          clear_table=args.clear_table, object_translation_world_m=[-args.closer_m, 0, 0],
                          official_benchmark_score=False, success_criterion="unchanged native bowl-on-plate task goal")
    print(json.dumps(provenance, indent=2), flush=True)
    if args.check:
        return
    args.output.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    torch.set_num_threads(8)
    started = time.perf_counter()
    policy = make_policy(config, env_cfg=environment_config, rename_map=saved["rename_map"]).eval()
    policy.requires_grad_(False)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(environment_config, config)
    kwargs = dict(environment_config.gym_kwargs)
    kwargs["task_ids"] = [args.task_id]
    grasp_episodes = []

    def reduced_scene(source, layout_seed=None):
        import numpy as np

        source.sim.forward()
        retained = {"akita_black_bowl_1", "plate_1"} if args.bowl_on_plate else {"akita_black_bowl_2", "akita_black_bowl_3"}
        removed = set(source.objects_dict) - retained if args.bowl_on_plate else {"plate_1", "akita_black_bowl_1"}
        if args.bowl_on_plate and source.parsed_problem["goal_state"] != [["on", "akita_black_bowl_1", "plate_1"]]:
            raise ValueError("Select an original task whose goal is black bowl 1 on plate 1")
        if not removed.issubset(source.objects_dict):
            raise ValueError("Expected original plate_1 and akita_black_bowl_1 in this scene")
        source_type = type(source)
        removed_fixtures = set(source.fixtures_dict) if args.clear_table else set()
        removed_regions = {name for name, region in source.parsed_problem["regions"].items()
                           if region["target"] in removed_fixtures}
        removed_references = removed | removed_fixtures | removed_regions
        translation = np.array([-args.closer_m, 0.0, 0.0])
        expected_positions = {name: source.sim.data.body_xpos[source.obj_body_id[name]].copy() + translation
                              for name in retained}
        if layout_seed is not None:
            base_id = source.sim.model.body_name2id(source.robots[0].robot_model.root_body)
            expected_positions = sample_bowl_plate_positions(expected_positions, source.sim.data.body_xpos[base_id],
                                                              layout_seed, centered=args.center_layout)

        class ReducedScene(source_type):
            def _load_fixtures_in_arena(self, arena):
                if args.clear_table:
                    self.parsed_problem["fixtures"] = {"table": ["main_table"]}
                    self.parsed_problem["regions"] = {
                        name: region for name, region in self.parsed_problem["regions"].items()
                        if name not in removed_regions}
                super()._load_fixtures_in_arena(arena)

            def _load_objects_in_arena(self, arena):
                self.parsed_problem["objects"] = {
                    category: [name for name in names if name not in removed]
                    for category, names in self.parsed_problem["objects"].items()}
                self.parsed_problem["initial_state"] = [state for state in self.parsed_problem["initial_state"]
                                                         if not any(name in removed_references for name in state)]
                if not args.bowl_on_plate:
                    self.parsed_problem["goal_state"] = []
                super()._load_objects_in_arena(arena)

            def _check_success(self):
                return super()._check_success() if args.bowl_on_plate else False

        ReducedScene.__name__ = source_type.__name__
        scene = ReducedScene(
            source.bddl_file_name, robots=["Panda"],
            controller_configs={key: value for key, value in source.robots[0].controller_config.items() if key != "sim"},
            initialization_noise=None, use_camera_obs=True, has_renderer=False, has_offscreen_renderer=True,
            camera_names=source.camera_names, camera_heights=source.camera_heights, camera_widths=source.camera_widths,
            control_freq=source.control_freq, horizon=source.horizon, ignore_done=True, hard_reset=False)
        try:
            scene.reset()
            for body_id, name in enumerate(scene.sim.model.body_names):
                source_id = source.sim.model.body_name2id(name)
                scene.sim.model.body_pos[body_id] = source.sim.model.body_pos[source_id]
                scene.sim.model.body_quat[body_id] = source.sim.model.body_quat[source_id]
            for name in scene.sim.model.joint_names:
                scene.sim.data.set_joint_qpos(name, source.sim.data.get_joint_qpos(name))
                scene.sim.data.set_joint_qvel(name, source.sim.data.get_joint_qvel(name))
            for obj in scene.objects_dict.values():
                pose = scene.sim.data.get_joint_qpos(obj.joints[0]).copy()
                pose[:3] += expected_positions[obj.name] - source.sim.data.body_xpos[source.obj_body_id[obj.name]]
                scene.sim.data.set_joint_qpos(obj.joints[0], pose)
            scene.sim.forward()
            scene.robots[0].controller.update(force=True)
            scene.robots[0].controller.reset_goal()
            scene.robots[0].gripper.current_action = source.robots[0].gripper.current_action.copy()
            scene._update_observables(force=True)
            expected = set(source.objects_dict) - removed
            assert set(scene.objects_dict) == expected == retained
            assert set(scene.fixtures_dict) == set(source.fixtures_dict) - removed_fixtures
            for name in (*scene.objects_dict, *scene.fixtures_dict):
                expected_position = source.sim.data.body_xpos[source.obj_body_id[name]].copy()
                if name in scene.objects_dict:
                    expected_position = expected_positions[name]
                np.testing.assert_allclose(scene.sim.data.body_xpos[scene.obj_body_id[name]],
                                           expected_position, atol=1e-7, rtol=0)
            np.testing.assert_allclose(scene.robots[0]._joint_positions, source.robots[0]._joint_positions, atol=1e-7)
            for name in scene.camera_names:
                camera_id = scene.sim.model.camera_name2id(name)
                source_id = source.sim.model.camera_name2id(name)
                np.testing.assert_allclose(scene.sim.data.cam_xpos[camera_id], source.sim.data.cam_xpos[source_id], atol=1e-7)
                np.testing.assert_allclose(scene.sim.data.cam_xmat[camera_id], source.sim.data.cam_xmat[source_id], atol=1e-7)
            return scene
        except BaseException:
            scene.close()
            raise

    class BlackBowlGrasp(gym.Wrapper):
        def __init__(self, environment):
            super().__init__(environment)
            self.original_instruction = environment.task_description
            environment.task_description = provenance.get("instruction_override", self.original_instruction)
            self.task_description = environment.task_description
            self.task = environment.task
            self.source_native = environment._env.env
            self.modified_native = None
            self.layout_metadata = None

        def reset(self, **reset_kwargs):
            episode_seed = reset_kwargs.get("seed")
            episode_seed = args.seed if episode_seed is None else int(episode_seed)
            layout_seed = args.layout_seed
            if args.vary_layout:
                layout_seed += episode_seed - args.seed
            if args.stack_two_bowls or args.bowl_on_plate:
                self.env._env.env = self.source_native
            observation, info = self.env.reset(**reset_kwargs)
            self.native = self.env._env.env
            if args.stack_two_bowls or args.bowl_on_plate:
                if self.modified_native is not None:
                    self.modified_native.close()
                self.modified_native = reduced_scene(self.native, layout_seed)
                self.env._env.env = self.native = self.modified_native
                observation = self.env._format_raw_obs(self.native._get_observations())
            self.bowls = {name: obj for name, obj in self.native.objects_dict.items()
                          if "black_bowl" in obj.category_name}
            if not self.bowls:
                raise ValueError("Original scene contains no black bowls")
            self.initial_positions = {name: self.native.sim.data.body_xpos[self.native.obj_body_id[name]].copy()
                                      for name in self.bowls}
            if args.export_scene:
                from projects.franka_stack_vla.bowl_plate_transfer import export_source
                export_source(self.native, self.task_description, args.output)
            self.holds = dict.fromkeys(self.bowls, 0)
            self.stack_holds = dict.fromkeys(self.bowls, 0)
            self.episode = {"original_instruction": self.original_instruction,
                            "instruction": self.task_description, "objects": list(self.native.objects_dict),
                            "fixtures": list(self.native.fixtures_dict),
                            "remaining_poses_and_cameras_verified": bool(args.stack_two_bowls or args.bowl_on_plate),
                            "goal": self.native.parsed_problem["goal_state"],
                            "object_translation_world_m": [-args.closer_m, 0, 0],
                            "object_positions": {name: self.native.sim.data.body_xpos[self.native.obj_body_id[name]].tolist()
                                                 for name in self.native.objects_dict},
                            "initial_positions": {name: position.tolist() for name, position in self.initial_positions.items()},
                            "steps": []}
            grasp_episodes.append(self.episode)
            if args.layout_seed is not None:
                base_id = self.native.sim.model.body_name2id(self.native.robots[0].robot_model.root_body)
                layout = {"layout_seed": layout_seed, "policy_seed": episode_seed, "reference_closer_m": args.closer_m,
                          "sampling_mode": "inward_center" if args.center_layout else "constant_robot_radius",
                          "object_positions": self.episode["object_positions"],
                          "reference_object_positions": {
                              name: (self.source_native.sim.data.body_xpos[self.source_native.obj_body_id[name]]
                                     + [-args.closer_m, 0, 0]).tolist() for name in self.native.objects_dict},
                          "robot_base_position": self.native.sim.data.body_xpos[base_id].tolist()}
                self.layout_metadata = layout
                layout_filename = f"layout_{episode_seed}.json" if args.vary_layout else "layout.json"
                (args.output / layout_filename).write_text(json.dumps(layout, indent=2))
            return observation, info

        def step(self, action):
            import numpy as np

            raw, _, _, _ = self.env._env.step(action)
            robot = self.native.robots[0]
            metrics = {}
            for name, bowl in self.bowls.items():
                position = self.native.sim.data.body_xpos[self.native.obj_body_id[name]]
                lift = float(position[2] - self.initial_positions[name][2])
                contact = all(self.native.check_contact(bowl, robot.gripper.important_geoms[side])
                              for side in ("left_finger", "right_finger"))
                self.holds[name] = self.holds[name] + 1 if contact and lift >= 0.05 else 0
                metrics[name] = {"lift_m": lift, "bilateral_contact": bool(contact),
                                 "hold_seconds": self.holds[name] / self.native.control_freq,
                                 "tcp_distance_m": float(np.linalg.norm(position - raw["robot0_eef_pos"]))}
            success = any(metric["hold_seconds"] >= 0.5 for metric in metrics.values())
            stack_metrics = {}
            if args.stack_two_bowls:
                released = bool(np.abs(raw["robot0_gripper_qpos"]).sum() > 0.07 and not any(
                    self.native.check_contact(bowl, robot.gripper.contact_geoms) for bowl in self.bowls.values()))
                stationary = all(np.linalg.norm(self.native.sim.data.get_body_xvelp(bowl.root_body)) < 0.02
                                 and np.linalg.norm(self.native.sim.data.get_body_xvelr(bowl.root_body)) < 0.3
                                 for bowl in self.bowls.values())
                withdrawn = all(metric["tcp_distance_m"] > 0.07 for metric in metrics.values())
                for top_name, top in self.bowls.items():
                    bottom_name = next(name for name in self.bowls if name != top_name)
                    native_on = bool(self.native._eval_predicate(["on", top_name, bottom_name]))
                    supported = bool(self.native.check_contact(self.bowls[bottom_name], "table_collision")
                                     and not self.native.check_contact(top, "table_collision"))
                    valid = native_on and supported and released and stationary and withdrawn
                    self.stack_holds[top_name] = self.stack_holds[top_name] + 1 if valid else 0
                    stack_metrics[top_name] = {"bottom": bottom_name, "native_on": native_on,
                                              "supported": supported, "released": released,
                                              "stationary": bool(stationary), "withdrawn": withdrawn,
                                              "hold_seconds": self.stack_holds[top_name] / self.native.control_freq}
                success = any(metric["hold_seconds"] >= 0.5 for metric in stack_metrics.values())
            if args.bowl_on_plate:
                success = bool(self.native._check_success())
            self.episode["steps"].append({"action": action.tolist(), "bowls": metrics,
                                          "stack": stack_metrics, "success": success})
            truncated = len(self.episode["steps"]) >= self.env._max_episode_steps
            info = {"is_success": success}
            if success or truncated:
                info["final_info"] = {"is_success": success}
            return self.env._format_raw_obs(raw), float(success), success, truncated, info

        def close(self):
            if self.modified_native is not None:
                self.source_native.close()
            super().close()

    def grasp_vector(environment_factories):
        from functools import partial

        def create_wrapped(factory):
            return BlackBowlGrasp(factory())

        return gym.vector.SyncVectorEnv([partial(create_wrapped, factory) for factory in environment_factories])

    environments = create_libero_envs(
        task=environment_config.task, n_envs=1, gym_kwargs=kwargs,
        camera_name=environment_config.camera_name, init_states=environment_config.init_states,
        env_cls=grasp_vector if args.grasp_black_bowl or args.stack_two_bowls or args.bowl_on_plate else gym.vector.SyncVectorEnv,
        control_mode=environment_config.control_mode,
        episode_length=environment_config.episode_length,
    )
    try:
        if args.rlinf_ppo_steps:
            from projects.franka_stack_vla.rlinf_bowl_ppo import train
            provenance.update(trained=True, residual=True, method="RLinf PPO", prior_frozen=True)
            (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            train(environments, policy, preprocessor, postprocessor, env_preprocessor,
                  args.resume_residual, args.output, args.rlinf_ppo_steps, args.seed, args.layout_seed)
            return
        if args.evaluate_pi05:
            from projects.franka_stack_vla.libero_bowl_residual import collect_teacher_corrections
            provenance.update(trained=False, residual=False, placement_only=True, policy="RLinf/RLinf-Pi05-LIBERO-SFT",
                              teacher_from_start=True)
            (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            collect_teacher_corrections(environments, policy, preprocessor, postprocessor, env_preprocessor,
                                        args.output, list(range(args.seed, args.seed + args.episodes)), 0,
                                        placement_only=True, teacher_from_start=True)
            return
        if args.evaluate_residual:
            from projects.franka_stack_vla.supervised_bowl_residual import evaluate_checkpoints
            provenance.update(trained=False, residual=not args.baseline_only, placement_only=True, teacher_used_in_evaluation=False)
            (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            evaluate_checkpoints(environments, policy, preprocessor, postprocessor, env_preprocessor,
                                 args.evaluate_residual, args.compare_residual,
                                 range(args.seed, args.seed + args.episodes), args.output, baseline_only=args.baseline_only,
                                 compare_baseline=args.compare_baseline, closer_m=args.closer_m,
                                 residual_start_step=args.residual_start_step,
                                 layout_seeds=([args.layout_seed + (offset if args.vary_layout else 0)
                                                for offset in range(args.episodes)] if args.layout_seed is not None else None))
            return
        if args.supervised_data:
            from projects.franka_stack_vla.supervised_bowl_residual import train_and_evaluate
            provenance.update(trained=True, residual=True, method="supervised teacher corrections", placement_only=True,
                              supervised_data=str(args.supervised_data.resolve()))
            (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            train_and_evaluate(environments, policy, preprocessor, postprocessor, env_preprocessor,
                               args.supervised_data, args.output, args.supervised_epochs,
                               resume_from=args.resume_residual, additional_sources=args.additional_corrections)
            return
        if args.teacher_corrections:
            from projects.franka_stack_vla.libero_bowl_residual import collect_teacher_corrections
            provenance.update(teacher_corrections=True, trained=False, takeover_step=args.takeover_step,
                              placement_only=args.placement_only, teacher_target=args.teacher_target,
                              residual_start_step=args.residual_start_step, layout_seed=args.layout_seed,
                              vary_layout=args.vary_layout, center_layout=args.center_layout)
            (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            collect_teacher_corrections(environments, policy, preprocessor, postprocessor, env_preprocessor,
                                        args.output, list(range(args.seed, args.seed + args.episodes)), args.takeover_step,
                                        target_successes=args.teacher_target, placement_only=args.placement_only,
                                        learner_checkpoint=args.learner_checkpoint, residual_start_step=args.residual_start_step)
            return
        if args.residual_steps:
            from projects.franka_stack_vla.libero_bowl_residual import train_residual
            provenance.update(trained=True, residual=True, training_steps=args.residual_steps)
            (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2))
            train_residual(environments, policy, preprocessor, postprocessor, env_preprocessor,
                           args.output, args.residual_steps, args.seed)
            return
        with torch.no_grad():
            results = eval_policy_all(
                envs=environments, policy=policy, env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor, preprocessor=preprocessor,
                postprocessor=postprocessor, n_episodes=args.episodes,
                max_episodes_rendered=args.episodes, videos_dir=args.output / "videos",
                start_seed=args.seed, max_parallel_tasks=1,
            )
        (args.output / "eval_info.json").write_text(json.dumps(results, indent=2) + "\n")
        if args.grasp_black_bowl or args.stack_two_bowls or args.bowl_on_plate:
            episodes = [episode for episode in grasp_episodes if episode["steps"]]
            (args.output / "grasp_trace.json").write_text(json.dumps(episodes) + "\n")
            metrics = [metric for episode in episodes for step in episode["steps"] for metric in step["bowls"].values()]
            grasp_summary = {"successes": sum(any(step["success"] for step in episode["steps"]) for episode in episodes),
                             "episodes": len(episodes), "max_lift_m": max(metric["lift_m"] for metric in metrics),
                             "bilateral_contact": any(metric["bilateral_contact"] for metric in metrics),
                             "max_hold_seconds": max(metric["hold_seconds"] for metric in metrics),
                             "closest_tcp_distance_m": min(metric["tcp_distance_m"] for metric in metrics),
                             "official_benchmark_score": False}
            if args.stack_two_bowls:
                stack_metrics = [metric for episode in episodes for step in episode["steps"] for metric in step["stack"].values()]
                grasp_summary.update(task="stack either bowl on the other", native_on=any(metric["native_on"] for metric in stack_metrics),
                                     max_stack_hold_seconds=max(metric["hold_seconds"] for metric in stack_metrics))
            if args.bowl_on_plate:
                grasp_summary.update(task="original bowl-on-plate goal with distractors removed", original_goal=True)
            (args.output / "grasp_summary.json").write_text(json.dumps(grasp_summary, indent=2) + "\n")
            print(json.dumps(grasp_summary, indent=2), flush=True)
        provenance["elapsed_seconds"] = time.perf_counter() - started
        (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(json.dumps(results["overall"], indent=2), flush=True)
    finally:
        close_envs(environments)


if __name__ == "__main__":
    main()