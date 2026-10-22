"""Evaluate a visual stacking policy using pre-reset physical success states."""

import argparse
import json
import os
from pathlib import Path

from run import compose_project


def summarize(records):
    if not records:
        raise ValueError("Evaluation requires completed episodes")
    return {
        "episodes": len(records),
        "success_rate": sum(record["success"] for record in records) / len(records),
        "workspace_failure_rate": sum(record["workspace_failure"] for record in records) / len(records),
        "mean_return": sum(record["return"] for record in records) / len(records),
        "mean_steps": sum(record["steps"] for record in records) / len(records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--config", choices=("train", "adaptive", "precision", "dino", "course"), default="adaptive")
    parser.add_argument("--stage", choices=("lift", "transport", "stack"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--diagnostics", action="store_true")
    options = parser.parse_args()
    if options.episodes < 1 or options.num_envs < 1:
        parser.error("episodes and num-envs must be positive")
    checkpoint = options.checkpoint.resolve(strict=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    overrides = ([f"env.train.init_params.training_stage={options.stage}"] if options.stage else [])
    config, _ = compose_project("my_robot", config=options.config, overrides=overrides)
    from projects.my_robot.env import (
        make_preview_simulation,
        workspace_failure, wrap_observations,
    )
    environment, owner = make_preview_simulation(
        config.env.train.init_params, options.num_envs, options.seed,
        config.env.train.max_episode_steps, headless=True, enable_cameras=True,
        viewer={"resolution": (1280, 720)} if options.video else None,
    )
    import torch
    from rlinf.models import get_model
    from tools.lift_ablation import checksum

    model = get_model(config.actor.model).to(environment.device).eval()
    reference_encoder = None
    if hasattr(model, "visual_backbone"):
        reference_encoder = {name: value.detach().cpu().clone()
                             for name, value in model.visual_backbone.state_dict().items()}
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if reference_encoder is not None:
        if not all(torch.equal(value, model.visual_backbone.state_dict()[name].cpu())
                   for name, value in reference_encoder.items()):
            raise ValueError("The frozen pretrained visual encoder changed during training")
    terminal_success = torch.zeros(options.num_envs, dtype=torch.bool, device=environment.device)
    terminal_workspace = terminal_success.clone()
    terminal_hold = torch.zeros_like(terminal_success, dtype=torch.float32)
    original_reset = environment._reset_idx

    diagnostics = {"steps": 0, "terms": {}, "max_blue_height_gain_m": 0.0,
                   "min_hand_blue_distance_m": None, "min_blue_goal_distance_m": None,
                   "closest_approach": None,
                   "policy_action_std": model.actor_logstd.exp().detach().cpu().tolist()}
    if options.diagnostics:
        original_compute = environment.reward_manager.compute

        def capture_rewards(dt):
            rewards = original_compute(dt)
            diagnostics["steps"] += options.num_envs
            for stage in ("reach", "lift", "align", "align_fine", "placed"):
                if stage not in environment.reward_manager.active_terms:
                    continue
                term = environment.reward_manager.get_term_cfg(stage)
                if term.weight == 0:
                    continue
                values = term.func(environment, **term.params)
                entry = diagnostics["terms"].setdefault(
                    stage, {"sum": 0.0, "positive_steps": 0, "reward_sum": 0.0}
                )
                entry["sum"] += float(values.sum())
                entry["positive_steps"] += int((values > 0).sum())
                entry["reward_sum"] += float(values.sum()) * term.weight * dt
            blue = environment.scene["cube_1"].data.root_pos_w
            red = environment.scene["cube_2"].data.root_pos_w
            hand = environment.scene["ee_frame"].data.target_pos_w[:, 0]
            height = blue[:, 2] - environment.scene.env_origins[:, 2] - 0.0203
            diagnostics["max_blue_height_gain_m"] = max(
                diagnostics["max_blue_height_gain_m"], float(height.max())
            )
            for key, distance in (
                ("min_hand_blue_distance_m", torch.linalg.vector_norm(hand - blue, dim=-1)),
                ("min_blue_goal_distance_m", torch.linalg.vector_norm(
                    blue - red - blue.new_tensor([0.0, 0.0, 0.0468]), dim=-1)),
            ):
                current = float(distance.min())
                if key == "min_hand_blue_distance_m" and (
                    diagnostics[key] is None or current < diagnostics[key]
                ):
                    index = int(distance.argmin())
                    robot = environment.scene["robot"]
                    finger_ids, _ = robot.find_joints(environment.cfg.gripper_joint_names)
                    diagnostics["closest_approach"] = {
                        "hand_pos_local": (hand[index] - environment.scene.env_origins[index]).tolist(),
                        "blue_pos_local": (blue[index] - environment.scene.env_origins[index]).tolist(),
                        "hand_quaternion_wxyz": environment.scene["ee_frame"].data.target_quat_w[index, 0].tolist(),
                        "finger_joint_pos": robot.data.joint_pos[index, finger_ids].tolist(),
                    }
                diagnostics[key] = current if diagnostics[key] is None else min(diagnostics[key], current)
            return rewards

        environment.reward_manager.compute = capture_rewards

    def capture_reset(env_ids):
        term = environment.cfg.terminations.success
        terminal_success[env_ids] = term.func(environment, **term.params)[env_ids]
        terminal_workspace[env_ids] = workspace_failure(environment)[env_ids]
        if hasattr(environment, "_stage_hold_steps"):
            terminal_hold[env_ids] = environment._stage_hold_steps[env_ids] * environment.step_dt
        original_reset(env_ids)

    environment._reset_idx = capture_reset
    environment.seed(options.seed)
    observations, _ = environment.reset()
    returns = torch.zeros(options.num_envs, device=environment.device)
    steps = torch.zeros(options.num_envs, dtype=torch.long, device=environment.device)
    records = []
    writer = None
    if options.video:
        import imageio.v2 as imageio

        options.video.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(30):
            environment.sim.render()
            frame = environment.render()
            if frame is not None and frame.size and frame.std() > 1:
                break
        else:
            raise RuntimeError("Renderer did not produce a nonblank frame")
        imageio.imwrite(options.video.with_suffix(".png"), frame)
        writer = imageio.get_writer(str(options.video), fps=round(1 / environment.step_dt),
                                   codec="libx264", macro_block_size=1)
    with torch.inference_mode():
        while len(records) < options.episodes:
            inputs = wrap_observations(observations)
            actions, _ = model.predict_action_batch(inputs, mode="eval")
            terminal_success.zero_()
            terminal_workspace.zero_()
            observations, rewards, terminated, truncated, _ = environment.step(actions[:, 0])
            returns += rewards
            steps += 1
            if writer is not None:
                writer.append_data(environment.render())
            for index in torch.where(terminated | truncated)[0].tolist():
                if len(records) < options.episodes:
                    records.append({
                        "success": bool(terminal_success[index]),
                        "workspace_failure": bool(terminal_workspace[index]),
                        "terminated": bool(terminated[index]),
                        "truncated": bool(truncated[index]),
                        "return": float(returns[index]),
                        "steps": int(steps[index]),
                        "terminal_stable_hold_s": float(terminal_hold[index]),
                    })
                returns[index] = 0
                steps[index] = 0
    if writer is not None:
        writer.close()
    report = {
        "task": "Workspace-Stack-Two-Cubes-Franka-v0",
        "recipe": options.config,
        "frozen_encoder_verified": reference_encoder is not None,
        "seed": options.seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checksum(checkpoint),
        "training_stage": getattr(environment.cfg, "training_stage", "stack"),
        "criterion": ("Blue height gain >= 0.04 m, both fingers contact blue with > 0.1 N, "
                  "blue speed < 0.1 m/s, hand distance < 0.06 m, inside workspace, "
                  "continuously >= 0.5 s; transport also requires goal error < 0.025 m"
                  if getattr(environment.cfg, "training_stage", "stack") != "stack" else
                  "Blue on red, low object velocities, gripper open, inside workspace; "
                  "measured before native reset"),
        "horizon": config.env.train.max_episode_steps,
        "video": str(options.video) if options.video else None,
        **summarize(records), "records": records,
    }
    if options.diagnostics:
        for entry in diagnostics["terms"].values():
            entry["mean"] = entry["sum"] / diagnostics["steps"]
        report["diagnostics"] = diagnostics
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)
    owner.close()


if __name__ == "__main__":
    main()