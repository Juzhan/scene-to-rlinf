"""Preview the actual simulator task without a trainer or parameter updates.

Each backend environment draws its own view window or video frames; this loop runs the policy."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

from shared.assembly.control import load_factory
from shared.backends.observations import checked_observations


def parse_options(arguments, operation):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--actions", choices=("hold", "zero", "random", "policy"),
                        help="hold (default): repeat the executor's initial action (its initial pose); zero: all-zero actions")
    parser.add_argument("--random-scale", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Use a saved project's frozen source (handled by run.py)")
    parser.add_argument("--allow-scene-change", action="store_true", help="Explicitly evaluate a changed scene with the original policy interface")
    parser.add_argument("--allow-shared-change", action="store_true", help="Explicitly accept and report shared framework changes")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true", help="Explicitly accept weights without saved interface metadata")
    parser.add_argument("--camera-eye", type=float, nargs=3)
    parser.add_argument("--camera-target", type=float, nargs=3)
    parser.add_argument("--camera", help="Show (view) or record the scene camera of this name, as its renderer draws it for the policy")
    parser.add_argument("--camera-render", action="store_true",
                        help="Draw the preview viewpoint with the scene cameras' renderer (Newton Warp or OVRTX) instead of the GL viewer")
    options, overrides = parser.parse_known_args(arguments)
    if not 1 <= options.num_envs <= 16:
        parser.error("--num-envs must be between 1 and 16")
    if options.steps < 0 or (operation == "record" and options.steps == 0):
        parser.error("--steps must be positive; only view accepts 0 for continuous preview")
    if not math.isfinite(options.random_scale) or not 0 < options.random_scale <= 1:
        parser.error("--random-scale must be finite and in (0, 1]")
    for vector in (options.camera_eye, options.camera_target):
        if vector is not None and not all(math.isfinite(value) for value in vector):
            parser.error("Camera coordinates must be finite")
    if options.checkpoint:
        options.checkpoint = options.checkpoint.expanduser().resolve()
        if not options.checkpoint.is_file():
            parser.error(f"Checkpoint not found: {options.checkpoint}")
        if options.actions not in (None, "policy"):
            parser.error("--checkpoint requires policy actions")
    options.actions = options.actions or ("policy" if options.checkpoint else "hold")
    return options, overrides


def checked_states(observations, count, dimension, adapter=None):
    import torch  # imported here so --help and option parsing work without the simulator group

    if dimension is None:
        return checked_observations(observations, count)
    packed = adapter(observations) if adapter is not None else observations if "states" in observations else None
    states = packed["states"] if packed is not None else observations["policy"]
    if isinstance(states, dict):
        from rlinf.envs.sim.isaaclab.utils import quat2axisangle_torch

        states = torch.cat((states["eef_pos"],
                            quat2axisangle_torch(states["eef_quat"][:, [1, 2, 3, 0]]),
                            states["gripper_pos"]), dim=1)
    if tuple(states.shape) != (count, dimension) or not torch.isfinite(states).all():
        raise ValueError(f"Expected finite policy observations [{count}, {dimension}]")
    return packed if packed is not None else {"states": states}


def update_environment_camera(native):
    """A terrain curriculum may relocate the selected environment at reset."""
    controller = getattr(native, "viewport_camera_controller", None)
    if controller is not None and controller.cfg.origin_type == "env":
        controller.update_view_to_env()


def preview(cfg, options, *, recording):
    import torch

    from shared.assembly.project import environment_factory

    parameters = cfg.env.train.init_params
    factory = parameters.get("preview_factory")
    # The project registers its constructor without the training framework (shared.assembly.project).
    constructor = load_factory(factory) if factory else environment_factory(parameters.id)
    # Registered assembly factories can keep their scene's camera setting in view mode.
    camera_override = None if constructor is not None and not factory and not recording else recording
    adapter_entry = parameters.get("observation_adapter")
    adapter = load_factory(adapter_entry) if adapter_entry else None
    if cfg.env.train.env_type != "isaaclab" or not (parameters.get("env_cfg") or constructor):
        raise ValueError("view/record require an Isaac task configuration or registered environment")
    if not recording and not os.environ.get("DISPLAY"):
        raise RuntimeError("No DISPLAY: use record or launch view from a desktop terminal")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(cfg.runner.logger.log_path) / "previews" / timestamp
    destination.mkdir(parents=True, exist_ok=False)
    report = {"task": parameters.id, "num_envs": options.num_envs, "steps": 0,
              "actions": options.actions, "resets": 0, "terminated": 0, "truncated": 0,
              "status": "failed", "task_success_evaluated": False, "training": False,
              "environment_closed": False, "video": None}
    if getattr(options, "experiment_report", None):
        report["project"] = options.experiment_report
    environment = native = writer = subscription = None
    previous_handler = None
    stop_requested = False

    def request_stop(*_):
        nonlocal stop_requested
        stop_requested = True

    try:
        viewer = {"resolution": (1280, 720)}
        if options.camera_eye is not None:
            viewer["eye"] = tuple(options.camera_eye)
        if options.camera_target is not None:
            viewer["lookat"] = tuple(options.camera_target)
        if options.camera:
            viewer["scene_cameras"] = True  # the scene's cameras render, whatever its enable_cameras says
        if options.camera_render:
            viewer["render"] = "camera"  # the backend draws the viewpoint through a camera (and errs if it cannot)
        if constructor is None:
            # A legacy task configured by a native env_cfg: Isaac Lab 2 through the RLinf subprocess.
            from shared.backends.isaac.legacy.state import make_simulation
            constructor = make_simulation
        environment, owner = constructor(
            parameters, options.num_envs, cfg.env.train.seed, cfg.env.train.max_episode_steps,
            headless=recording, enable_cameras=camera_override, viewer=viewer,
        )
        native = getattr(environment, "native", environment)
        if getattr(options, "experiment_report", None) and getattr(options, "experiment_module", None):
            from importlib import import_module
            from shared.experiments import verify_resolved_interface
            from tools.inspect_scene import native_interface
            actual = native_interface(native)
            report["project"]["resolved_interface_verified"] = verify_resolved_interface(
                import_module(options.experiment_module), actual,
                snapshot=options.experiment_report.get("checkpoint_snapshot"))
            report["interface"] = actual
        application = owner.application
        if application is not None:
            native.sim._disable_app_control_on_stop_handle = True
        previous_handler = signal.signal(signal.SIGINT, request_stop)
        if not recording and application is not None:
            from omni.appwindow import get_default_app_window

            subscription = get_default_app_window().get_window_close_event_stream().create_subscription_to_pop(request_stop)
        if options.camera and not hasattr(native, "camera_image"):
            raise ValueError("--camera needs a backend that renders scene cameras for preview (isaaclab3)")
        if options.camera_render and not hasattr(native, "show_preview"):
            raise ValueError("--camera-render needs a backend that renders the viewpoint through a camera (isaaclab3)")
        if options.camera:
            native.camera_image(options.camera)  # an unknown name fails before the run starts

        def camera_frame():
            image = native.camera_image(options.camera)
            # Small sensor images are scaled up by whole pixels for the video, keeping their pixels visible.
            factor = max(1, 640 // image.shape[1])
            return image.repeat(factor, axis=0).repeat(factor, axis=1) if factor > 1 else image

        dimension = native.action_manager.total_action_dim
        if dimension != parameters.action_dim:
            raise ValueError(f"Configured action_dim={parameters.action_dim}, actual={dimension}")
        observations, _ = environment.reset()
        update_environment_camera(native)
        states = checked_states(observations, options.num_envs, parameters.obs_dim, adapter)
        report.update(obs_shape={key: list(value.shape) for key,value in states.items()}, action_shape=[options.num_envs, dimension],
                      step_dt=native.step_dt, seed=cfg.env.train.seed)
        model = None
        if options.actions == "policy":
            try:
                from rlinf.models import get_model
            except ModuleNotFoundError:
                raise ValueError("Policy preview loads a training model: install the train group "
                                 "(tools/setup_env.py --with train) or preview with --actions hold") from None

            torch.manual_seed(cfg.actor.seed)
            model = get_model(cfg.actor.model).to(environment.device).eval()
            checkpoint = options.checkpoint or cfg.actor.model.get("initial_checkpoint")
            if checkpoint:
                checkpoint = Path(checkpoint).expanduser().resolve()
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
                with checkpoint.open("rb") as stream:
                    report["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
                report["checkpoint"] = str(checkpoint)
            else:
                print("WARNING: policy preview uses an untrained model", flush=True)
        generator = torch.Generator(device=environment.device).manual_seed(cfg.env.train.seed)
        # The assembly control chain (shared/assembly/policy_runtime): its initial executor action, when the
        # executor is the trained component whose actions the preview stands in for.
        chain = getattr(native, "_assembly_control_chain", None)
        if chain is not None and chain.selected != "executor":
            chain = None
        if recording:
            import imageio.v2 as imageio

            for _ in range(30):
                if application is not None:
                    native.sim.render()
                frame = camera_frame() if options.camera else native.render()
                if frame is not None and frame.size and frame.std() > 1:
                    break
            else:
                if not options.camera:
                    raise RuntimeError("Renderer did not produce a nonblank frame")
                # A scene camera may well see one flat colour (a wall, the inside of its own housing).
                print(f"WARNING: camera {options.camera} sees a single colour; check where it is mounted and where it looks",
                      flush=True)
            imageio.imwrite(destination / "first_frame.png", frame)
            video = destination / "preview.mp4"
            writer = imageio.get_writer(str(video), fps=1 / native.step_dt,
                                       codec="libx264", macro_block_size=1)
            report["video"] = str(video)
        print(f"Preview ready: actions={options.actions}, envs={options.num_envs}, output={destination}", flush=True)
        with torch.inference_mode():
            while (application is None or application.is_running()) and not stop_requested:
                if options.steps and report["steps"] >= options.steps:
                    break
                if application is not None and native.sim.is_stopped():
                    break
                if application is not None and not native.sim.is_playing():
                    native.sim.render()
                    continue
                started = time.monotonic()
                if model is not None:
                    chunks, _ = model.predict_action_batch(states, mode="eval")
                    if tuple(chunks.shape) != (options.num_envs, 1, dimension):
                        raise ValueError("Policy must return [num_envs, 1, action_dim]")
                    actions = chunks[:, 0]
                elif options.actions == "random":
                    actions = (2 * torch.rand((options.num_envs, dimension), device=environment.device,
                                             generator=generator) - 1) * options.random_scale
                elif options.actions == "hold" and getattr(chain, "initial_output", None) is not None:
                    actions = chain.initial_output.clone()  # the executor's initial action, every step
                else:
                    actions = torch.zeros((options.num_envs, dimension), device=environment.device)
                if not torch.isfinite(actions).all():
                    raise ValueError("Non-finite preview actions")
                try:
                    observations, rewards, terminated, truncated, step_info = environment.step(actions)
                    if bool((terminated | truncated).any()):
                        update_environment_camera(native)
                except (ReferenceError, RuntimeError):
                    if not recording and (stop_requested or application is None or not application.is_running()):
                        break
                    raise
                dones = terminated | truncated
                states = checked_states(observations, options.num_envs, parameters.obs_dim, adapter)
                if not torch.isfinite(rewards).all():
                    raise ValueError("Non-finite preview rewards")
                report["steps"] += 1
                report["resets"] += int((terminated | truncated).sum())
                report["terminated"] += int(terminated.sum())
                report["truncated"] += int(truncated.sum())
                if options.camera and not recording:
                    native.show_image(native.camera_image(options.camera))
                elif options.camera_render and not recording:
                    native.show_preview()
                if writer is not None:
                    frame = camera_frame() if options.camera else native.render()
                    if frame is None or not frame.size or (frame.std() <= 1 and not options.camera):
                        raise RuntimeError("Renderer returned a blank frame")
                    writer.append_data(frame)
                # A backend that draws its own window reports when the user closed it.
                if getattr(native, "window_closed", False):
                    stop_requested = True
                if not recording:
                    remaining = native.step_dt - (time.monotonic() - started)
                    if remaining > 0:
                        time.sleep(remaining)
        report["status"] = "completed" if options.steps and report["steps"] == options.steps else "stopped"
        report["simulated_seconds"] = report["steps"] * native.step_dt
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        try:
            if writer is not None:
                writer.close()
        finally:
            subscription = None
            if previous_handler is not None:
                signal.signal(signal.SIGINT, previous_handler)
            if environment is not None:
                if hasattr(native, "_rgb_annotator"):
                    native._rgb_annotator.detach()
                    del native._rgb_annotator
                if hasattr(native, "_render_product"):
                    native._render_product.destroy()
                    del native._render_product
                environment.close()
                report["environment_closed"] = True
            with (destination / "report.json").open("x") as stream:
                json.dump(report, stream, indent=2)
            print(json.dumps(report, indent=2), flush=True)
            print(f"Report: {destination / 'report.json'}", flush=True)
    return report


def main(cfg, options, *, recording):
    status = 0
    try:
        preview(cfg, options, recording=recording)
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
