"""Run FrankaSim visual SAC independently or through the workspace project."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party/RLinf"
MODEL = ROOT / "third_party/RLinf-ResNet10-pretrained"
os.environ.setdefault("EMBODIED_PATH", str(UPSTREAM / "examples/embodiment"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
sys.path.insert(0, str(UPSTREAM))


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def make_config(output, epochs, workspace=False, recipe="train", resume_dir=None):
    from hydra import compose, initialize_config_dir
    from run import _configure_checkpoint_interval

    if workspace:
        from run import compose_project

        overrides = [f"runner.max_epochs={epochs}",
                     f"runner.logger.log_path={Path(output).resolve() / 'runs'}"]
        if resume_dir is not None:
            overrides.append(f"runner.resume_dir={Path(resume_dir).resolve()}")
        config, _ = compose_project("frankasim", recipe, overrides=overrides)
        return config, overrides
    if resume_dir is not None:
        raise ValueError("Resume is supported only for workspace experiments")
    overrides = [
        f"runner.max_epochs={epochs}", "runner.save_interval=${runner.max_epochs}",
        "algorithm.replay_buffer.cache_size=256",
        "algorithm.replay_buffer.sample_window_size=256",
        f"runner.logger.log_path={Path(output).resolve() / 'runs'}",
        f"actor.model.model_path={MODEL}", f"rollout.model.model_path={MODEL}",
        "+env.train.render_size=[720,1280]", "+env.eval.render_size=[720,1280]",
        "+env.train.resize_to=[128,128]", "+env.eval.resize_to=[128,128]",
    ]
    with initialize_config_dir(version_base="1.1", config_dir=str(UPSTREAM / "examples/embodiment/config")):
        config = compose(config_name="frankasim_sac_cnn_async", overrides=overrides)
    _configure_checkpoint_interval(config)
    return config, overrides


def stable_grasp(height_gain, speed, hand_distance, left_force, right_force):
    return (height_gain >= 0.2 and speed < 0.1 and hand_distance < 0.06
            and left_force > 0.1 and right_force > 0.1)


def accepted(reports):
    return (len(reports) == 2 and {report["seed"] for report in reports} == {43, 44}
            and all(report["episodes"] >= 32 and report["success_rate"] >= 0.8
                    and report["held_0_5s_rate"] >= 0.8 and report["frozen_encoder_verified"]
                    for report in reports))


def evaluate(config, checkpoint, output, seed=43, episodes=32, video=None):
    import gym
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    import torch
    from rlinf.envs.sim.frankasim.frankasim_env import FrankaSimEnv
    from rlinf.models import get_model

    np.random.seed(seed)
    torch.manual_seed(seed)
    environment = FrankaSimEnv(config.env.eval, 1, seed, 1, None)
    native = environment.envs[0]
    base = native.unwrapped
    model = get_model(config.actor.model).cuda().eval()
    reference = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                 if ".resnet_backbone." in name}
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    frozen = bool(reference) and all(torch.equal(value, model.state_dict()[name].cpu())
                                     for name, value in reference.items())
    if not frozen:
        raise ValueError("Official pretrained ResNet backbones changed")
    block_id = base.model.geom("block").id
    block_body = int(base.model.geom_bodyid[block_id])
    writer = imageio.get_writer(str(video), fps=50, codec="libx264", macro_block_size=1) if video else None
    records = []
    try:
        for episode in range(episodes):
            raw, _ = native.reset(seed=seed + episode)
            success = False
            hold_steps = 0
            max_hold = 0.0
            max_height = 0.0
            episode_return = 0.0
            for step in range(config.env.eval.max_episode_steps):
                packed = environment._collate_obs([environment._wrap_obs(raw)])
                actions, _ = model.predict_action_batch(packed, mode="eval")
                raw, reward, terminated, truncated, info = native.step(actions[0, 0].cpu().numpy())
                success |= bool(info.get("success", False))
                episode_return += float(reward)
                height = float(base.data.sensor("block_pos").data[2] - base._z_init)
                velocity = np.zeros(6)
                mujoco.mj_objectVelocity(base.model, base.data, mujoco.mjtObj.mjOBJ_BODY,
                                        block_body, velocity, 0)
                distance = float(np.linalg.norm(base.data.sensor("block_pos").data
                                               - base.data.sensor("2f85/pinch_pos").data))
                forces = {"left": 0.0, "right": 0.0}
                for contact_index in range(base.data.ncon):
                    contact = base.data.contact[contact_index]
                    if block_id not in (contact.geom1, contact.geom2):
                        continue
                    other = contact.geom2 if contact.geom1 == block_id else contact.geom1
                    body_name = base.model.body(int(base.model.geom_bodyid[other])).name
                    for side in forces:
                        if body_name.startswith(side):
                            force = np.zeros(6)
                            mujoco.mj_contactForce(base.model, base.data, contact_index, force)
                            forces[side] += abs(float(force[0]))
                valid = stable_grasp(height, float(np.linalg.norm(velocity[3:])), distance,
                                     forces["left"], forces["right"])
                hold_steps = hold_steps + 1 if valid else 0
                max_hold = max(max_hold, hold_steps * base.control_dt)
                max_height = max(max_height, height)
                if writer is not None and episode < 2:
                    frame = raw["images"]["front"]
                    writer.append_data(frame)
                    if episode == 0 and step == 0:
                        imageio.imwrite(Path(video).with_suffix(".png"), frame)
                if terminated or truncated:
                    break
            records.append({"success": success, "held_0_5s": max_hold >= 0.5 - 1e-6,
                            "max_stable_hold_s": max_hold, "max_height_gain_m": max_height,
                            "return": episode_return, "steps": step + 1})
    finally:
        if writer is not None:
            writer.close()
        environment.close()
    report = {"seed": seed, "episodes": episodes, "checkpoint": str(Path(checkpoint).resolve()),
              "checkpoint_sha256": checksum(checkpoint), "frozen_encoder_verified": frozen,
              "success_rate": sum(record["success"] for record in records) / episodes,
              "held_0_5s_rate": sum(record["held_0_5s"] for record in records) / episodes,
              "mean_return": sum(record["return"] for record in records) / episodes,
              "criterion": "Official lift >=20 cm; additional continuous 0.5s bilateral contact "
                           ">0.1N each, blue speed <0.1m/s, pinch distance <6cm",
              "camera_size": [720, 1280], "network_size": [128, 128], "records": records}
    write_json(output, report)
    return report


def ablation_comparison(reports, baseline):
    episodes = sum(report["episodes"] for report in reports)
    baseline_episodes = sum(report["episodes"] for report in baseline)
    success = sum(report["success_rate"] * report["episodes"] for report in reports) / episodes
    holding = sum(report["held_0_5s_rate"] * report["episodes"] for report in reports) / episodes
    reference_success = sum(report["success_rate"] * report["episodes"] for report in baseline) / baseline_episodes
    reference_holding = sum(report["held_0_5s_rate"] * report["episodes"] for report in baseline) / baseline_episodes
    valid = (len(reports) == 2 and {report["seed"] for report in reports} == {43, 44}
         and all(report["episodes"] >= 32 and report["frozen_encoder_verified"]
             and report["success_rate"] >= 0.8 for report in reports))
    return {"passed": valid and success >= reference_success - 0.05 and holding >= reference_holding,
        "success_rate": success, "held_0_5s_rate": holding,
        "baseline_success_rate": reference_success, "baseline_held_0_5s_rate": reference_holding,
        "success_tolerance": 0.05, "caveat": "Engineering gate on fixed evaluation seeds, not statistical superiority"}


def prepare(output, epochs, workspace=False, recipe="train", resume_dir=None):
    from omegaconf import OmegaConf

    output = Path(output).resolve()
    config, overrides = make_config(output, epochs, workspace, recipe, resume_dir)
    resume = None
    if resume_dir is not None:
        from rlinf.utils.checkpoint import parse_global_step_from_checkpoint_path

        checkpoint_directory = Path(resume_dir).resolve()
        start_step = parse_global_step_from_checkpoint_path(str(checkpoint_directory))
        actor_directory = checkpoint_directory / "actor"
        if not (actor_directory / "model_state_dict/full_weights.pt").is_file() or not (
                actor_directory / "dcp_checkpoint/.metadata").is_file():
            raise ValueError("Resume requires a complete native model and optimizer checkpoint")
        if epochs <= start_step:
            raise ValueError("Total epochs must exceed the resumed global step")
        resume = {"directory": str(checkpoint_directory), "global_step": start_step,
                  "checkpoint_sha256": {str(path): checksum(path) for path in actor_directory.rglob("*")
                                         if path.is_file()},
                  "additional_epochs": epochs - start_step,
                  "additional_transitions": (epochs - start_step) * config.env.train.max_steps_per_rollout_epoch,
                  "caveat": "Native model/optimizer resume; environment restarts, not bitwise simulation continuation"}
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), UPSTREAM / "examples/embodiment/train_async.py",
               UPSTREAM / "examples/embodiment/config/frankasim_sac_cnn_async.yaml",
               UPSTREAM / "rlinf/envs/sim/frankasim/frankasim_env.py",
               UPSTREAM / "rlinf/models/embodiment/cnn_policy/cnn_policy.py",
               UPSTREAM / "rlinf/models/embodiment/modules/resnet_utils.py"]
    sources.extend((ROOT / "third_party/serl-frankasim/franka_sim/franka_sim").rglob("*.py"))
    if workspace:
        sources.extend([ROOT / "run.py", ROOT / "projects/frankasim/project.py",
                        ROOT / "projects/frankasim/train.yaml"])
        sources.extend((ROOT / "projects/frankasim").glob("*.py"))
        if recipe != "train":
            sources.append(ROOT / f"projects/frankasim/{recipe}.yaml")
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "epochs": epochs,
            "recipe": f"projects/frankasim/{recipe}.yaml" if workspace else "frankasim_sac_cnn_async",
            "workspace_project": "frankasim" if workspace else None,
            "workspace_config": recipe if workspace else None,
            "phase_two": "Workspace reproduction authorized by user; fresh training" if workspace
                         else "Only after independent acceptance",
            "config": OmegaConf.to_container(config, resolve=True), "overrides": overrides,
            "source_sha256": {str(path.relative_to(ROOT)): checksum(path) for path in sources},
            "model_sha256": checksum(MODEL / "resnet10_pretrained.pt"),
            "differences": ["720p rendering resized to 128x128 instead of native 128 rendering",
                            "Replay window/cache 256 trajectories instead of 20000; original eagerly allocates >250GB per image field",
                            "checkpoint save interval 420; independent physical evaluation"],
            "acceptance": {"seeds": [43, 44], "episodes_each": 32,
                           "success_rate_each": 0.8, "held_0_5s_rate_each": 0.8},
            "caveat": "GPU wall time and capability are unverified until training and independent evaluation."}
    if resume is not None:
        plan["resume"] = resume
        plan["phase_two"] = "User-authorized continuation from full native checkpoint"
    if workspace and recipe == "ppo":
        baseline_path = ROOT / "outputs/frankasim/workspace_20261007/status.json"
        baseline = json.loads(baseline_path.read_text())
        plan["ablation"] = {"factor": "SAC recipe to PPO with same task and ResNet10",
                            "baseline": str(baseline_path), "baseline_sha256": checksum(baseline_path),
                            "baseline_results": baseline["results"],
                            "planned_transitions": epochs * config.env.train.max_steps_per_rollout_epoch,
                            "baseline_transitions": 271872,
                            "changes": ["PPO/GAE/clipping and value head replace SAC/Q heads/replay",
                                        "SAC critic DrQ is not applied to PPO likelihoods; deterministic visual preprocessing retained",
                                        "Torch compilation/CUDA graphs disabled for PPO runtime compatibility",
                                        "Bounded-action adapter stores raw samples for exact PPO likelihood ratios"],
                            "ppo_stability": "Gaussian KL target 0.01; pooling dropout disabled for exact train/eval likelihoods; per-round policy sync; native PPO learnable state-independent std initialized at exp(-0.5)",
                            "next_stage": "DINO only after comparison gate passes; not started automatically"}
        if resume is not None:
            plan["ablation"]["factor"] = "User-authorized PPO training budget extension; same resumed PPO recipe"
            plan["ablation"]["budget_caveat"] = "Extended PPO interaction budget, not a matched-interaction SAC comparison"
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"state": "prepared"})
    return plan


def execute(output):
    from omegaconf import OmegaConf

    output = Path(output).resolve()
    plan = json.loads((output / "plan.json").read_text())
    workspace = plan.get("workspace_project") == "frankasim"
    recipe = plan.get("workspace_config") or "train"
    config, overrides = make_config(output, plan["epochs"], workspace, recipe,
                                    plan.get("resume", {}).get("directory"))
    if OmegaConf.to_container(config, resolve=True) != plan["config"]:
        raise ValueError("Frozen reference configuration changed")
    for name, expected in plan["source_sha256"].items():
        if checksum(ROOT / name) != expected:
            raise ValueError(f"Frozen reference source changed: {name}")
    if checksum(MODEL / "resnet10_pretrained.pt") != plan["model_sha256"]:
        raise ValueError("Pretrained weights changed")
    for name, expected in plan.get("resume", {}).get("checkpoint_sha256", {}).items():
        if checksum(name) != expected:
            raise ValueError(f"Frozen resume checkpoint changed: {name}")
    status = {"state": "running", "phase": "workspace_training" if workspace else "official_training",
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", status)
    environment = dict(os.environ, PYTHONPATH=str(UPSTREAM), PYTHONUNBUFFERED="1",
                       MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    try:
        if workspace:
            command = [sys.executable, str(ROOT / "run.py"), "train", "frankasim", "--config", recipe, *overrides]
        else:
            command = [sys.executable, str(UPSTREAM / "examples/embodiment/train_async.py"),
                       "--config-name", "frankasim_sac_cnn_async", *overrides]
        with (output / "train.log").open("w") as log:
            subprocess.run(command, cwd=UPSTREAM, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        candidates = list((output / "runs").rglob(f"global_step_{plan['epochs']}/actor/model_state_dict/full_weights.pt"))
        if len(candidates) != 1:
            raise ValueError("Expected one complete final native checkpoint")
        checkpoint = candidates[0]
        status.update(phase="independent_evaluation", checkpoint=str(checkpoint))
        write_json(output / "status.json", status)
        reports = []
        for seed in (43, 44):
            destination = output / f"evaluation_seed{seed}.json"
            command = [sys.executable, "-m", "tools.frankasim_reference", "evaluate",
                       "--output", str(destination), "--checkpoint", str(checkpoint),
                       "--seed", str(seed), "--episodes", "32"]
            if workspace:
                command.extend(["--workspace", "--config", recipe])
            if seed == 43:
                command.extend(["--video", str(output / "replay.mp4")])
            environment["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(UPSTREAM)))
            with (output / f"evaluation_seed{seed}.log").open("w") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            reports.append(json.loads(destination.read_text()))
        passed = accepted(reports)
        official_passed = all(report["success_rate"] >= 0.8 for report in reports)
        next_phase = ("completed" if workspace else "awaiting_workspace_replication") if passed else "awaiting_analysis"
        status.update(state="accepted" if passed else "acceptance_failed", phase=next_phase,
                  official_accepted=official_passed, results=reports,
                      finished_at=datetime.now(timezone.utc).isoformat())
        if "ablation" in plan:
            comparison = ablation_comparison(reports, plan["ablation"]["baseline_results"])
            status.update(ablation_comparison=comparison, full_physical_accepted=passed)
            if comparison["passed"]:
                status.update(state="comparison_passed", phase="awaiting_dino_ablation")
        write_json(output / "status.json", status)
    except Exception as error:
        status.update(state="failed", error=str(error))
        write_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "execute", "evaluate"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15000)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-dir", type=Path, help="Full native global_step directory; epochs is the total target")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--workspace", action="store_true", help="Use projects.frankasim and run.py train")
    parser.add_argument("--config", default="train", choices=("train", "ppo"))
    options = parser.parse_args()
    if options.epochs < 1 or options.episodes < 1:
        parser.error("epochs and episodes must be positive")
    if options.command == "prepare":
        prepare(options.output, options.epochs, options.workspace, options.config, options.resume_dir)
    elif options.command == "execute":
        execute(options.output)
    else:
        if options.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        config, _ = make_config(options.output.parent, options.epochs, options.workspace, options.config)
        evaluate(config, options.checkpoint, options.output, options.seed, options.episodes, options.video)


if __name__ == "__main__":
    main()