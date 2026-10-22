"""Independent physical staircase routes using the registered Assembly Go2 environment."""

import argparse
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import traceback

from run import compose_project


_SUITE_DEFAULTS = {
    "seeds": [43, 44], "levels": [0, 4], "directions": [0, 90, 180, 270],
    "body_headings": [0], "steps": 600, "speed": 0.45, "minimum_success": 0.8,
}


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_options(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_name", nargs="?", help="Project used by run.py assess")
    parser.add_argument("--project")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="Use a saved project's frozen configuration and code")
    parser.add_argument("--allow-scene-change", action="store_true")
    parser.add_argument("--allow-shared-change", action="store_true")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument("--suite", type=Path, help="Defaults to the selected project's evaluation.yaml")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--directions", type=float, nargs="+")
    parser.add_argument("--body-headings", type=float, nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--minimum-success", type=float)
    parser.add_argument("--video", type=Path, help="Record only the first route of the first seed")
    options = parser.parse_args(arguments)
    if options.project and options.project_name and options.project != options.project_name:
        parser.error("The positional project and --project must match")
    options.project = options.project or options.project_name
    if options.snapshot and options.recipe:
        parser.error("--snapshot uses frozen configuration; omit --recipe")
    return options


def load_suite(options, project_directory):
    from omegaconf import OmegaConf

    explicit = options.suite is not None
    path = (options.suite or Path(project_directory) / "evaluation.yaml").expanduser().resolve()
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True) if path.is_file() else {}
    if not isinstance(raw, dict):
        raise ValueError("Evaluation suite must be a mapping")
    if explicit and not path.is_file():
        raise ValueError(f"Evaluation suite does not exist: {path}")
    unknown = raw.keys() - (_SUITE_DEFAULTS.keys() | {"schema_version", "type"})
    if unknown or raw.get("schema_version", 1) != 1 or raw.get("type", "go2_stairs_routes") != "go2_stairs_routes":
        raise ValueError(f"Unsupported staircase evaluation suite: {path}; unknown fields: {sorted(unknown)}")
    values = {**_SUITE_DEFAULTS, **{key: value for key, value in raw.items() if key in _SUITE_DEFAULTS}}
    overrides = {key: getattr(options, key) for key in _SUITE_DEFAULTS if getattr(options, key) is not None}
    values.update(overrides)
    for name in ("seeds", "levels", "directions", "body_headings"):
        items = values[name]
        if not isinstance(items, list) or not items or len(set(items)) != len(items):
            raise ValueError(f"Evaluation {name} must be a nonempty list without duplicates")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in items):
            raise ValueError(f"Evaluation {name} must contain finite numbers")
    if any(type(item) is not int or item < 0 for name in ("seeds", "levels") for item in values[name]):
        raise ValueError("Evaluation seeds and levels must be nonnegative integers")
    if type(values["steps"]) is not int or values["steps"] < 1:
        raise ValueError("Evaluation steps must be a positive integer")
    for name in ("speed", "minimum_success"):
        if isinstance(values[name], bool) or not isinstance(values[name], (int, float)) or not 0 < values[name] <= 1:
            raise ValueError(f"Evaluation {name} must be in (0, 1]")
    for key, value in values.items():
        setattr(options, key, value)
    return {"path": str(path) if path.is_file() else None,
            "sha256": file_sha256(path) if path.is_file() else None,
            "settings": values, "overrides": overrides}


def load_project(options):
    """Select source before registration so frozen and current factories cannot mix."""
    from omegaconf import OmegaConf
    from shared.experiments import checkpoint_compatibility, load_snapshot, prepare_replay, read_manifest
    from tools.inspect_scene import resolve_model_dimensions

    if options.snapshot:
        _, manifest = read_manifest(options.snapshot)
        if options.project and options.project != manifest["project"]:
            raise ValueError(f"Snapshot belongs to {manifest['project']}, not {options.project}")
        options.project = manifest["project"]
        cfg, module = load_snapshot(options.snapshot, register=False, allow_shared_change=options.allow_shared_change)
        report = checkpoint_compatibility(
            cfg, module, options.checkpoint, allow_scene_change=options.allow_scene_change,
            allow_shared_change=options.allow_shared_change, allow_legacy_checkpoint=options.allow_legacy_checkpoint)
    else:
        options.project = options.project or "assembly_go2"
        cfg, module = compose_project(options.project, register=False)
        if options.recipe:
            cfg = OmegaConf.load(options.recipe)
        resolve_model_dimensions(cfg, module)
        cfg, module, report = prepare_replay(
            cfg, module, checkpoint=options.checkpoint,
            allow_scene_change=options.allow_scene_change, allow_shared_change=options.allow_shared_change,
            allow_legacy_checkpoint=options.allow_legacy_checkpoint)
    return cfg, module, report


def route_cases(terrains, levels, directions, body_headings):
    cases = []
    for (name, column), level, direction, body_heading, outward in product(
        terrains, levels, directions, body_headings, (False, True)
    ):
        angle = math.radians(direction)
        unit = (math.cos(angle), math.sin(angle))
        scale = max(abs(unit[0]), abs(unit[1]))
        start_radius, goal_radius = ((1.1, 3.4) if outward else (-3.4, -1.1))
        cases.append({
            "terrain": name, "column": column, "level": level,
            "world_direction_deg": direction, "body_heading_offset_deg": body_heading,
            "outward": outward,
            "start_xy": [value * start_radius / scale for value in unit],
            "goal_xy": [value * goal_radius / scale for value in unit],
            "unit_xy": list(unit), "yaw": angle + math.radians(body_heading),
        })
    return cases


def summarize(records):
    if not records:
        raise ValueError("Evaluation must contain at least one route")
    groups = {}
    for record in records:
        key = (record["terrain"], record["level"], record["motion"],
               record["world_direction_deg"], record["body_heading_offset_deg"])
        groups.setdefault(key, []).append(record)
    return {
        "episodes": len(records),
        "success_rate": sum(record["success"] for record in records) / len(records),
        "fall_rate": sum(record["fell"] for record in records) / len(records),
        "by_motion": {motion: {"episodes": len(values),
                                "success_rate": sum(value["success"] for value in values) / len(values),
                                "fall_rate": sum(value["fell"] for value in values) / len(values)}
                      for motion in ("up", "down") if (values := [record for record in records if record["motion"] == motion])},
        "by_seed": [{"seed": seed, "episodes": len(values),
                      "success_rate": sum(value["success"] for value in values) / len(values)}
                     for seed in sorted({record.get("seed", 0) for record in records})
                     if (values := [record for record in records if record.get("seed", 0) == seed])],
        "by_case": [dict(zip(
            ("terrain", "level", "motion", "world_direction_deg", "body_heading_offset_deg"), key
        ), episodes=len(values), success_rate=sum(value["success"] for value in values) / len(values),
            fall_rate=sum(value["fell"] for value in values) / len(values))
                    for key, values in groups.items()],
    }


def acceptance(records, seeds, minimum_success):
    """Neither easier downhill routes nor another seed can hide an uphill failure."""
    checks = []
    for seed in seeds:
        for motion in (None, "up", "down"):
            subset = [record for record in records if record["seed"] == seed and
                      (motion is None or record["motion"] == motion)]
            rate = sum(record["success"] for record in subset) / len(subset) if subset else None
            checks.append({"seed": seed, "motion": motion or "all", "episodes": len(subset),
                           "success_rate": rate, "passed": rate is not None and rate >= minimum_success})
    return {"accepted": bool(checks) and all(check["passed"] for check in checks),
            "minimum_success": minimum_success, "checks": checks}


def physical_evaluation_state(robot, scanner):
    """Read only the physical tensors used to score staircase routes.

    The native environment freezes these tensors before its in-step reset.
    """
    return {
        "root_pos_w": robot.data.root_pos_w,
        "heading_w": robot.data.heading_w,
        "projected_gravity_b": robot.data.projected_gravity_b,
        "root_lin_vel_w": robot.data.root_lin_vel_w,
        "ray_hits_w": scanner.data.ray_hits_w,
    }


def evaluation_step_state(live_state, dones, info):
    """Use the final physical state for rows reset by the native step."""
    if not info.get("autoreset", False) or not dones.any():
        return live_state
    terminal = info.get("terminal_state")
    if terminal is None:
        raise RuntimeError("Native autoreset evaluation requires a terminal physical state snapshot")
    result = {}
    for name, value in live_state.items():
        previous = terminal[name]
        if previous.shape != value.shape:
            raise ValueError(f"Terminal physical state {name} must contain the full environment batch")
        result[name] = value.clone()
        result[name][dones] = previous[dones]
    return result


def main(arguments=None):
    options = parse_options(arguments)
    checkpoint = options.checkpoint.resolve(strict=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    config, module, replay_report = load_project(options)
    project = module.project
    from omegaconf import OmegaConf

    suite = load_suite(options, project.scene_config.parent)
    module.register()
    scene_path = Path(config.env.train.init_params.get("scene_config", project.scene_config))
    settings = OmegaConf.load(config.env.train.init_params.get(
        "scene_config", project.scene_config
    ))
    generator = next(entity for entity in settings.entities if entity.name == "terrain").parameters.terrain_generator
    if not all(0 <= value < generator.num_rows for value in options.levels):
        raise ValueError("Evaluation levels must be valid terrain rows")
    fraction = 0.0
    terrains = []
    total = sum(item.parameters.proportion for item in generator.sub_terrains.values())
    for name, item in generator.sub_terrains.items():
        terrains.append((name, int((fraction + item.parameters.proportion / total / 2) * generator.num_cols)))
        fraction += item.parameters.proportion / total
    cases = route_cases(terrains, options.levels, options.directions, options.body_headings)
    environment, owner = project.make_environment(
        config.env.train.init_params, len(cases), options.seeds[0], options.steps,
        headless=True, enable_cameras=bool(options.video),
        viewer={"resolution": (1280, 720)} if options.video else None,
    )
    import torch
    from isaaclab.utils.math import quat_from_euler_xyz, quat_apply_inverse
    from isaaclab.utils.warp import raycast_mesh
    from rlinf.models import get_model

    native = environment.native
    device = environment.device
    robot = native.scene["robot"]
    terrain = native.scene["terrain"]
    scanner = native.scene["height_scanner"]
    native.terminal_state_fn = lambda: physical_evaluation_state(robot, scanner)
    from tools.inspect_scene import native_interface, apply_dimensions
    interface = native_interface(native)
    from shared.experiments import verify_resolved_interface
    if replay_report:
        replay_report["resolved_interface_verified"] = verify_resolved_interface(
            module, interface, snapshot=replay_report.get("checkpoint_snapshot"))
    apply_dimensions(config, interface)
    model = get_model(config.actor.model).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    ids = torch.arange(len(cases), device=device)
    levels = torch.tensor([case["level"] for case in cases], device=device)
    columns = torch.tensor([case["column"] for case in cases], device=device)
    unit = torch.tensor([case["unit_xy"] for case in cases], device=device)
    yaw = torch.tensor([case["yaw"] for case in cases], device=device)
    start_local = torch.tensor([case["start_xy"] for case in cases], device=device)
    goal_local = torch.tensor([case["goal_xy"] for case in cases], device=device)
    origins = terrain.terrain_origins[levels, columns]
    start = origins.clone()
    goal = origins.clone()
    start[:, :2] += start_local
    goal[:, :2] += goal_local
    rays = torch.cat((start, goal))
    rays[:, 2] += 20
    down = torch.zeros_like(rays)
    down[:, 2] = -1
    ground, _, _, _ = raycast_mesh(rays, down, scanner.meshes[scanner.cfg.mesh_prim_paths[0]])
    if not torch.isfinite(ground).all():
        raise RuntimeError("Evaluation routes must be covered by the actual terrain mesh")
    start[:, 2] = ground[:len(cases), 2]
    goal[:, 2] = ground[len(cases):, 2]
    height_delta = goal[:, 2] - start[:, 2]
    if not (height_delta.abs() > 0.08).all():
        raise RuntimeError("Evaluation routes must cross real stairs, not a flat platform")
    route_length = torch.linalg.vector_norm(goal[:, :2] - start[:, :2], dim=-1)
    records = []
    writer = None
    video_frames = 0
    try:
        with torch.inference_mode():
            for seed in options.seeds:
                environment.reset(seed=seed)
                terrain.terrain_levels[:] = levels
                terrain.terrain_types[:] = columns
                terrain.env_origins[:] = origins
                state = robot.data.default_root_state.clone()
                state[:, :3] = start
                state[:, 2] += robot.cfg.init_state.pos[2]
                state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
                state[:, 7:] = 0
                robot.write_root_state_to_sim(state, env_ids=ids)
                native.scene.reset(ids)
                native.scene.write_data_to_sim()
                native.sim.forward()
                native.scene.update(native.step_dt)
                if options.video and seed == options.seeds[0]:
                    import imageio.v2 as imageio
                    options.video.parent.mkdir(parents=True, exist_ok=True)
                    writer = imageio.get_writer(str(options.video), fps=1 / native.step_dt,
                                               codec="libx264", macro_block_size=1)
                command = native.command_manager.get_term("base_velocity")
                if writer is not None:
                    # Keep the whole physical route in a fixed side view. A
                    # following camera makes small stair rises hard to inspect.
                    command.set_debug_vis(False)
                    center = ((start[0] + goal[0]) * 0.5).cpu().tolist()
                    dx, dy = cases[0]["unit_xy"]
                    top = max(float(start[0, 2]), float(goal[0, 2]))
                    native.sim.set_camera_view(
                        eye=(center[0] + 4.5 * dy - 0.6 * dx,
                             center[1] - 4.5 * dx - 0.6 * dy, top + 2.0),
                        target=(center[0], center[1], center[2] + 0.22),
                    )
                    # The first call attaches the RGB annotator. Warm the
                    # renderer without physics steps before recording frames.
                    native.render()
                    for _ in range(3):
                        native.sim.render()
                    native.render()
                command.time_left[:] = (options.steps + 1) * native.step_dt + 1
                command.is_standing_env[:] = False
                command.heading_target[:] = yaw
                done = torch.zeros(len(cases), dtype=torch.bool, device=device)
                success = done.clone()
                fell = done.clone()
                hold = torch.zeros(len(cases), dtype=torch.long, device=device)
                steps = torch.zeros_like(hold)
                returns = torch.zeros(len(cases), device=device)
                final_progress = returns.clone()
                final_height = returns.clone()
                final_lateral = returns.clone()
                initial_z = None
                for step in range(options.steps):
                    offset = robot.data.root_pos_w[:, :2] - start[:, :2]
                    progress = (offset * unit).sum(dim=-1)
                    world_velocity = torch.zeros(len(cases), 3, device=device)
                    # A fixed proportional waypoint controller supplies velocity commands;
                    # the learned policy alone controls all twelve robot joints.
                    error = goal[:, :2] - robot.data.root_pos_w[:, :2]
                    desired = 1.5 * error
                    desired *= (options.speed / torch.linalg.vector_norm(desired, dim=-1).clamp_min(options.speed))[:, None]
                    world_velocity[:, :2] = desired * (~done)[:, None]
                    command.vel_command_b[:, :2] = quat_apply_inverse(robot.data.root_quat_w, world_velocity)[:, :2]
                    command._update_command()
                    states = native.observation_manager.compute()["states"]
                    actions, _ = model.predict_action_batch({"states": states}, mode="eval")
                    _, rewards, terminated, truncated, info = environment.step(actions[:, 0])
                    physical = evaluation_step_state(
                        physical_evaluation_state(robot, scanner), terminated | truncated, info
                    )
                    position = physical["root_pos_w"]
                    if not torch.isfinite(position).all():
                        raise RuntimeError("Non-finite physical evaluation state")
                    if initial_z is None:
                        initial_z = position[:, 2].clone()
                    offset = position[:, :2] - start[:, :2]
                    progress = (offset * unit).sum(dim=-1)
                    lateral = (offset[:, 0] * unit[:, 1] - offset[:, 1] * unit[:, 0]).abs()
                    height = position[:, 2] - initial_z
                    signed_height = height * height_delta.sign()
                    hits = physical["ray_hits_w"]
                    nearest = (hits[:, :, :2] - position[:, None, :2]).square().sum(dim=-1).argmin(dim=-1)
                    clearance = position[:, 2] - hits[ids, nearest, 2]
                    ground_change = hits[ids, nearest, 2] - start[:, 2]
                    yaw_error = torch.atan2(torch.sin(physical["heading_w"] - yaw), torch.cos(physical["heading_w"] - yaw))
                    at_goal = (
                        (torch.linalg.vector_norm(position[:, :2] - goal[:, :2], dim=-1) <= 0.35)
                        & (progress >= route_length - 0.35) & (lateral <= 0.35)
                        & (ground_change * height_delta.sign() >= height_delta.abs() - 0.06)
                        & (physical["projected_gravity_b"][:, 2] < -0.85)
                        & (clearance >= 0.18) & (clearance <= 0.55)
                        & (torch.linalg.vector_norm(physical["root_lin_vel_w"], dim=-1) < 0.35)
                        & (yaw_error.abs() <= math.pi / 6) & ~terminated
                    )
                    active = ~done
                    steps += active
                    returns += rewards * active
                    final_progress[active] = progress[active]
                    final_height[active] = height[active]
                    final_lateral[active] = lateral[active]
                    hold = torch.where(active & at_goal, hold + 1, torch.zeros_like(hold))
                    passed = hold >= math.ceil(0.5 / native.step_dt)
                    success |= active & passed
                    fell |= active & terminated
                    done |= passed | terminated | truncated
                    if writer is not None and not bool((terminated | truncated)[0]):
                        frame = native.render()
                        if frame is None or not frame.size:
                            raise RuntimeError("No rendered frame during route evaluation")
                        writer.append_data(frame)
                        video_frames += 1
                    if writer is not None and bool(done[0]):
                        # Native autoreset has already moved a finished robot;
                        # never append the next episode to this route's video.
                        writer.close()
                        writer = None
                    if done.all():
                        break
                for index, case in enumerate(cases):
                    records.append(dict(case, seed=seed, success=bool(success[index]), fell=bool(fell[index]),
                                        steps=int(steps[index]), return_value=float(returns[index]),
                                        motion="up" if height_delta[index] > 0 else "down",
                                        expected_height_delta_m=float(height_delta[index]),
                                        route_length_m=float(route_length[index]),
                                        progress_m=float(final_progress[index]),
                                        lateral_error_m=float(final_lateral[index]), height_delta_m=float(final_height[index])))
                print(f"seed={seed} episodes={len(cases)} success={int(success.sum())} falls={int(fell.sum())}", flush=True)
                if writer is not None:
                    writer.close()
                    writer = None
        report = {
            "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "project": options.project, "interface": interface,
            "scene": {"path": str(scene_path), "sha256": file_sha256(scene_path)},
            "project_replay": replay_report, "suite": suite,
            "criterion": "Cross the entire real stair route; goal and lateral error <=0.35m; measured support-ground elevation change >= actual route ground change minus0.06m; upright, clearance0.18-0.55m, speed<0.35m/s, heading error<=30deg, no base-contact fall, hold0.5s. A fixed proportional waypoint controller supplies velocity commands, the learned policy controls joints.",
            "command_controller": {"type": "fixed_proportional_waypoint", "position_gain": 1.5,
                                   "max_linear_speed_m_s": options.speed, "heading_target": "route yaw",
                                   "heading_control": "native base_velocity command manager",
                                   "policy_controls": "all twelve robot joints", "learned_navigation": False},
            "seeds": options.seeds, "command_speed_m_s": options.speed,
            "summary": summarize(records), "records": records,
            "video": {"path": str(options.video) if options.video else None, "frames": video_frames,
                      "seed": options.seeds[0], "route": cases[0], "camera": "fixed route side view"},
        }
        if replay_report and replay_report.get("snapshot"):
            report["snapshot_manifest_sha256"] = file_sha256(Path(replay_report["snapshot"]) / "manifest.json")
        if replay_report and replay_report.get("selected_snapshot"):
            report["selected_snapshot_manifest_sha256"] = file_sha256(Path(replay_report["selected_snapshot"]) / "manifest.json")
        report["acceptance"] = acceptance(records, options.seeds, options.minimum_success)
        report["minimum_success"] = options.minimum_success
        report["accepted"] = report["acceptance"]["accepted"]
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        if options.train_log:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            events = EventAccumulator(str(options.train_log), size_guidance={"scalars": 0}).Reload()
            figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            for axis, tag in zip(axes.flat, (
                "rollout/rewards", "env/episode_len", "train/actor/gaussian_kl", "train/actor/lr"
            )):
                if tag in events.Tags()["scalars"]:
                    values = events.Scalars(tag)
                    axis.plot([value.step for value in values], [value.value for value in values], linewidth=1)
                else:
                    axis.text(0.5, 0.5, "Metric not logged", ha="center")
                axis.set_title(tag)
                axis.set_xlabel("PPO update")
                axis.grid(alpha=0.25)
            figure.savefig(options.output.with_name("curves.png"), dpi=140)
            plt.close(figure)
        print(json.dumps({key: value for key, value in report["summary"].items() if key != "by_case"}), flush=True)
    except BaseException:
        if writer is not None:
            writer.close()
        traceback.print_exc()
        os._exit(1)
    owner.close()


if __name__ == "__main__":
    main()
