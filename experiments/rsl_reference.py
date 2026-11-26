"""Run the pinned official RSL-RL entry point with bounded simulator shutdown."""

import os
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "third_party" / "isaaclab_reference"


def measure_task(object_position, target_position, end_effector_position):
    import torch

    height = object_position[:, 2]
    goal_error = torch.linalg.vector_norm(object_position - target_position, dim=-1)
    hand_error = torch.linalg.vector_norm(object_position - end_effector_position, dim=-1)
    lifted = height > 0.10
    return height, goal_error, hand_error, lifted, lifted & (goal_error < 0.05)


def rlinf_state_dict(native_weights):
    if "std" not in native_weights or (native_weights["std"] <= 0).any():
        raise ValueError("Expected a positive scalar-parameterized RSL action standard deviation")
    converted = {"actor_logstd": native_weights["std"].log()}
    for name, value in native_weights.items():
        if name.startswith("actor."):
            converted[name] = value
        elif name.startswith("critic."):
            converted[name.replace("critic.", "value_head.", 1)] = value
        elif name != "std":
            raise ValueError(f"Unsupported native model parameter: {name}")
    return converted


def evaluate(arguments):
    sys.path.insert(0, str(REFERENCE / "python"))
    sys.path.insert(0, str(ROOT))
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trainer", choices=("rsl", "rlinf"), default="rsl")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    options = parser.parse_args(arguments)
    options.enable_cameras = options.video is not None
    options.checkpoint = options.checkpoint.resolve(strict=True)
    if options.num_envs < 1 or options.episodes < 1:
        parser.error("num_envs and episodes must be positive")
    application = AppLauncher(options).app

    import gymnasium as gym
    import torch
    import isaaclab_tasks
    from isaaclab.utils.math import quat_apply_inverse
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    task = "Isaac-Lift-Cube-Franka-v0"
    config = parse_env_cfg(task, device=options.device, num_envs=options.num_envs)
    config.seed = options.seed
    if options.video:
        config.viewer.origin_type = "env"
        config.viewer.env_index = 0
        config.viewer.eye = (1.5, 1.1, 0.9)
        config.viewer.lookat = (0.45, 0.0, 0.2)
        config.viewer.resolution = (1280, 720)
        config.scene.clone_in_fabric = False
    raw_environment = gym.make(task, cfg=config,
                               render_mode="rgb_array" if options.video else None)
    environment = RslRlVecEnvWrapper(raw_environment)
    base = environment.unwrapped
    if options.trainer == "rsl":
        agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(environment, agent.to_dict(), log_dir=None, device=base.device)
        runner.load(str(options.checkpoint))
        policy = runner.get_inference_policy(device=base.device)
    else:
        from run import compose_project
        from rlinf.models import get_model

        project, _ = compose_project("franka_lift")
        model = get_model(project.actor.model).to(base.device).eval()
        model.load_state_dict(torch.load(options.checkpoint, map_location="cpu", weights_only=True),
                              strict=True)

        def policy(observations):
            actions, _ = model.predict_action_batch({"states": observations["policy"]}, mode="eval")
            return actions[:, 0]

    def measurements():
        robot = base.scene["robot"]
        position = quat_apply_inverse(robot.data.root_quat_w,
                                      base.scene["object"].data.root_pos_w - robot.data.root_pos_w)
        hand = quat_apply_inverse(robot.data.root_quat_w,
                                  base.scene["ee_frame"].data.target_pos_w[:, 0] - robot.data.root_pos_w)
        target = base.command_manager.get_command("object_pose")[:, :3]
        return measure_task(position, target, hand)

    terminal_measurements = None
    original_reset = base._reset_idx

    def capture_reset(env_ids):
        nonlocal terminal_measurements
        terminal_measurements = tuple(value.clone() for value in measurements())
        original_reset(env_ids)

    base._reset_idx = capture_reset
    device = base.device
    lifted_once = torch.zeros(options.num_envs, dtype=torch.bool, device=device)
    goal_once = lifted_once.clone()
    held_steps = torch.zeros(options.num_envs, dtype=torch.long, device=device)
    max_held = held_steps.clone()
    episode_steps = held_steps.clone()
    returns = torch.zeros(options.num_envs, device=device)
    max_height = returns.clone()
    min_goal_error = torch.full_like(returns, float("inf"))
    min_hand_error = min_goal_error.clone()
    records = []
    writer = None
    if options.video:
        import imageio.v2 as imageio

        options.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(options.video), fps=round(1 / base.step_dt),
                                   codec="libx264", macro_block_size=1)
    environment.seed(options.seed)
    observations, _ = environment.reset()
    if writer is not None:
        for _ in range(30):
            base.sim.render()
            frame = base.render()
            if frame is not None and frame.std() > 1:
                break
        else:
            raise RuntimeError("Video renderer did not produce a nonblank initial frame")
    maximum_steps = base.max_episode_length * (options.episodes // options.num_envs + 2)
    try:
        with torch.inference_mode():
            for step in range(maximum_steps):
                terminal_measurements = None
                observations, rewards, dones, _ = environment.step(policy(observations))
                dones = dones.bool()
                values = measurements()
                if terminal_measurements is not None:
                    values = tuple(torch.where(dones, terminal, current)
                                   for terminal, current in zip(terminal_measurements, values))
                height, goal_error, hand_error, lifted, reached = values
                lifted_once |= lifted
                goal_once |= reached
                held_steps = torch.where(reached, held_steps + 1, 0)
                max_held = torch.maximum(max_held, held_steps)
                max_height = torch.maximum(max_height, height)
                min_goal_error = torch.minimum(min_goal_error, goal_error)
                min_hand_error = torch.minimum(min_hand_error, hand_error)
                episode_steps += 1
                returns += rewards
                if writer is not None:
                    frame = base.render()
                    if frame is None or frame.std() <= 1:
                        raise RuntimeError("Video renderer produced a blank frame")
                    if step == 0:
                        imageio.imwrite(options.video.with_suffix(".png"), frame)
                    writer.append_data(frame)
                for env_index in dones.nonzero(as_tuple=False).flatten().tolist():
                    records.append({"lifted_10cm_once": lifted_once[env_index].item(),
                                    "goal_reached_once": goal_once[env_index].item(),
                                    "goal_held_seconds": max_held[env_index].item() * base.step_dt,
                                    "maximum_height": max_height[env_index].item(),
                                    "minimum_goal_error": min_goal_error[env_index].item(),
                                    "minimum_hand_error": min_hand_error[env_index].item(),
                                    "return": returns[env_index].item(),
                                    "steps": episode_steps[env_index].item()})
                lifted_once[dones] = False
                goal_once[dones] = False
                held_steps[dones] = 0
                max_held[dones] = 0
                episode_steps[dones] = 0
                returns[dones] = 0
                max_height[dones] = 0
                min_goal_error[dones] = float("inf")
                min_hand_error[dones] = float("inf")
                if len(records) >= options.episodes:
                    break
    finally:
        if writer is not None:
            writer.close()
    records = records[:options.episodes]
    if len(records) != options.episodes:
        raise RuntimeError("Evaluation did not complete the requested episode count")
    with options.checkpoint.open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"task": task, "trainer": options.trainer, "seed": options.seed,
              "checkpoint": str(options.checkpoint), "checkpoint_sha256": checkpoint_hash,
              "episodes": len(records), "goal_tolerance_m": 0.05,
              "lift_height_m": 0.10, "video": str(options.video) if options.video else None,
              "lift_rate": sum(item["lifted_10cm_once"] for item in records) / len(records),
              "goal_rate": sum(item["goal_reached_once"] for item in records) / len(records),
              "held_0_5s_rate": sum(item["goal_held_seconds"] >= 0.5 for item in records) / len(records),
              "mean_return": sum(item["return"] for item in records) / len(records),
              "records": records}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)
    os._exit(0)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
        return evaluate(sys.argv[2:])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REFERENCE / "python")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    command = [sys.executable, str(REFERENCE / "rsl_rl" / "train.py"), *sys.argv[1:]]
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    completed = False
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            if line.startswith("Training time:"):
                completed = True
                break
        if completed:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            print("Official training completed; final checkpoint saved; simulator stopped.", flush=True)
            return 0
        return process.wait() or 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


if __name__ == "__main__":
    raise SystemExit(main())