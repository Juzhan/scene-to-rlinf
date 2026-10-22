"""Prepare and execute a frozen, paired Franka Lift learning-rate experiment."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run import ROOT, compose_project


CONDITIONS = ("adaptive", "fixed")
SOURCES = (
    "run.py",
    "run.sh",
    "configs/state_ppo.yaml",
    "projects/cartpole/policy.py",
    "projects/franka_lift/policy.py",
    "projects/franka_lift/env.py",
    "projects/franka_lift/project.py",
    "projects/franka_lift/train.yaml",
    "projects/franka_lift/ablation.yaml",
    "shared/isaac_state.py",
    "tools/lift_ablation.py",
    "tools/rsl_reference.py",
    "third_party/RLinf/rlinf/algorithms/utils.py",
    "third_party/RLinf/rlinf/algorithms/losses.py",
    "third_party/RLinf/rlinf/algorithms/advantages.py",
    "third_party/RLinf/rlinf/workers/actor/embodied_fsdp_actor_worker.py",
    "third_party/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py",
    "third_party/RLinf/rlinf/workers/env/env_worker.py",
    "third_party/RLinf/rlinf/data/schema/embodied_types.py",
    "third_party/RLinf/rlinf/data/schema/embodied_trajectory.py",
    "projects/my_robot/policy.py",
    "projects/my_robot/env.py",
    "projects/my_robot/project.py",
    "projects/my_robot/train.yaml",
    "projects/my_robot/adaptive.yaml",
    "projects/my_robot/precision.yaml",
    "projects/my_robot/dino.yaml",
    "projects/my_robot/course.yaml",
    "projects/my_robot/dino_policy.py",
    "tools/stack_evaluation.py",
)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def flatten_config(config, prefix=""):
    result = {}
    for name, value in config.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            result.update(flatten_config(value, key))
        else:
            result[key] = value
    return result


def build_conditions(output, epochs=1500, seed=42):
    from omegaconf import OmegaConf

    output = Path(output).resolve()
    conditions = {}
    for name in CONDITIONS:
        overrides = [
            f"runner.max_epochs={epochs}",
            "runner.resume_dir=null",
            f"runner.logger.experiment_name={name}",
            f"runner.logger.log_path={output / 'runs' / name}",
            f"actor.seed={seed}",
            f"env.train.seed={seed}",
            f"env.eval.seed={seed}",
            f"actor.model.initial_checkpoint={output / 'initial_weights.pt'}",
            f"algorithm.adaptive_kl_target={'0.01' if name == 'adaptive' else 'null'}",
        ]
        config, _ = compose_project("franka_lift", config="ablation", overrides=overrides)
        conditions[name] = {
            "overrides": overrides,
            "config": OmegaConf.to_container(config, resolve=True),
        }
    left = flatten_config(conditions["adaptive"]["config"])
    right = flatten_config(conditions["fixed"]["config"])
    differences = sorted(key for key in left.keys() | right.keys()
                         if left.get(key) != right.get(key))
    expected = sorted(("algorithm.adaptive_kl_target", "runner.logger.experiment_name",
                       "runner.logger.log_path"))
    if differences != expected:
        raise ValueError(f"Unexpected experiment differences: {differences}")
    return conditions, differences


def build_builtin_conditions(output, epochs=1500, seed=42, stack_epochs=1000):
    from omegaconf import OmegaConf

    output = Path(output).resolve()
    conditions = {}
    for name, project, recipe, iterations, peak in (
        ("lift_cosine_1e3", "franka_lift", "ablation", epochs, 0.001),
        ("stack_adaptive", "my_robot", "adaptive", stack_epochs, 0.0001),
        ("lift_cosine_3e3", "franka_lift", "ablation", epochs, 0.003),
    ):
        initial = "stack_initial_weights.pt" if project == "my_robot" else "initial_weights.pt"
        overrides = [
            f"runner.max_epochs={iterations}", "runner.resume_dir=null",
            f"runner.logger.experiment_name={name}",
            f"runner.logger.log_path={output / 'runs' / name}",
            f"actor.seed={seed}", f"env.train.seed={seed}", f"env.eval.seed={seed}",
            f"actor.model.initial_checkpoint={output / initial}",
        ]
        if project == "franka_lift":
            overrides.extend([
                "algorithm.adaptive_kl_target=null", "actor.optim.lr_scheduler=cosine",
                f"actor.optim.lr={peak}", f"actor.optim.value_lr={peak}",
                "actor.optim.min_lr_rate=0.1", "actor.optim.lr_warmup_steps=0",
            ])
        config, _ = compose_project(project, config=recipe, overrides=overrides)
        conditions[name] = {
            "project": project, "recipe": recipe, "iterations": iterations,
            "episodes_per_evaluation_seed": 32 if project == "my_robot" else 128,
            "overrides": overrides, "config": OmegaConf.to_container(config, resolve=True),
        }
    left = flatten_config(conditions["lift_cosine_1e3"]["config"])
    right = flatten_config(conditions["lift_cosine_3e3"]["config"])
    differences = sorted(key for key in left.keys() | right.keys()
                         if left.get(key) != right.get(key))
    expected = sorted(("actor.optim.lr", "actor.optim.value_lr",
                       "runner.logger.experiment_name", "runner.logger.log_path"))
    if differences != expected:
        raise ValueError(f"Unexpected cosine experiment differences: {differences}")
    return conditions, differences


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_stack_conditions(output, epochs=1000, seed=42, precision=False, dino=False, course=False):
    from omegaconf import OmegaConf

    output = Path(output).resolve()
    if course:
        conditions = {}
        previous = output / "initial_weights.pt"
        descriptions = {
            "lift": "Lift the blue block from the table and hold it steadily between both fingers.",
            "transport": "Lift the blue block and hold it steadily above the red block.",
            "stack": "Stack the blue block on the red block and release it.",
        }
        for stage, description in descriptions.items():
            name = f"visual_{stage}"
            overrides = [
                f"runner.max_epochs={epochs}", "runner.resume_dir=null",
                f"runner.logger.experiment_name={name}",
                f"runner.logger.log_path={output / 'runs' / name}",
                f"actor.seed={seed}", f"env.train.seed={seed}", f"env.eval.seed={seed}",
                f"actor.model.initial_checkpoint={previous}",
                f"env.train.init_params.training_stage={stage}",
                f"env.train.init_params.task_description='{description}'",
            ]
            config, _ = compose_project("my_robot", config="course", overrides=overrides)
            conditions[name] = {
                "project": "my_robot", "recipe": "course", "stage": stage,
                "iterations": epochs, "episodes_per_evaluation_seed": 32,
                "overrides": overrides, "config": OmegaConf.to_container(config, resolve=True),
                "acceptance": {"episodes": 64, "each_seed_success_rate": 0.8,
                               "each_seed_max_workspace_failure_rate": 0.05},
            }
            previous = (output / "runs" / name / name / "checkpoints"
                        / f"global_step_{epochs}" / "actor" / "model_state_dict" / "full_weights.pt")
        return conditions, []
    name = "stack_dino_v4" if dino else "stack_precision_v3" if precision else "stack_reward_v2"
    recipe = "dino" if dino else "precision" if precision else "adaptive"
    overrides = [
        f"runner.max_epochs={epochs}", "runner.resume_dir=null",
        f"runner.logger.experiment_name={name}",
        f"runner.logger.log_path={output / 'runs' / name}",
        f"actor.seed={seed}", f"env.train.seed={seed}", f"env.eval.seed={seed}",
        f"actor.model.initial_checkpoint={output / 'initial_weights.pt'}",
    ]
    config, _ = compose_project("my_robot", config=recipe, overrides=overrides)
    return {name: {"project": "my_robot", "recipe": recipe, "iterations": epochs,
                   "episodes_per_evaluation_seed": 32, "overrides": overrides,
                   "config": OmegaConf.to_container(config, resolve=True)}}, []


def prepare_stack(output, initial, epochs=1000, seed=42, precision=False, dino=False, course=False):
    dino = dino or course
    initial = Path(initial).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(initial, output / "initial_weights.pt")
    conditions, differences = build_stack_conditions(
        output, epochs, seed, precision=precision, dino=dino, course=course,
    )
    transform = None
    encoder_sources = {}
    encoder_info = None
    if precision or dino:
        import torch

        weights = torch.load(output / "initial_weights.pt", map_location="cpu", weights_only=True)
        condition = next(iter(conditions.values()))
        deviation = condition["config"]["actor"]["model"]["initial_std"]
        if dino:
            from rlinf.models import get_model

            config, _ = compose_project("my_robot", config="dino")
            torch.manual_seed(seed)
            model = get_model(config.actor.model)
            reference = weights
            weights = {name: value.cpu() for name, value in model.state_dict().items()}
            retained = ("state_encoder.", "actor.", "value_head.")
            for name in weights:
                if name.startswith(retained):
                    weights[name] = reference[name].cpu().clone()
            snapshot = Path(config.actor.model.visual_checkpoint)
            encoder_sources = {str(snapshot / name): checksum(snapshot / name)
                               for name in ("config.json", "model.safetensors")}
            encoder_info = {"model": "facebook/dinov2-small", "revision": snapshot.name,
                            "frozen": True, "input_size": list(config.actor.model.visual_input_size),
                            "spatial_grid": [4, 7], "cache_features_for_ppo": True,
                            "parameters": sum(value.numel() for value in model.visual_backbone.parameters()),
                            "trainable_parameters": sum(value.numel() for value in model.parameters()
                                                        if value.requires_grad),
                            "retained_reference_prefixes": list(retained)}
        weights["actor_logstd"] = weights["actor_logstd"].new_tensor(deviation).log()
        torch.save(weights, output / "initial_weights.pt")
        transform = {"parameter": "actor_logstd", "initial_std": deviation}
    plan = {
        "suite": "stack_dino" if dino else "stack_precision" if precision else "stack_reward",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_seed": seed, "iterations": epochs, "evaluation_seeds": [43, 44],
        "episodes_per_evaluation_seed": 32,
        "reference_initial": str(initial),
        "reference_initial_sha256": checksum(initial),
        "initial_weight_transform": transform,
        "visual_encoder": encoder_info,
        "encoder_source_sha256": encoder_sources,
        "initial_weights_sha256": checksum(output / "initial_weights.pt"),
        "source_sha256": {name: checksum(ROOT / name) for name in SOURCES},
        "allowed_differences": differences, "order": list(conditions), "conditions": conditions,
        "reward_design": {"grasp_removed": True, "height_deadband_m": 0.005,
                          "lift_saturation_gain_m": 0.04, "goal_gate_gain_m": 0.025,
                          "dense_rate_weights": {"reach": 1, "lift": 15, "align": 16,
                                                 "align_fine": 5, "placed": 12},
                          "success_bonus": 200, "step_dt": 0.05,
                          "nonterminal_dense_discounted_bound_gamma_098": 92.5},
        "caveat": ("One visual training seed, small sample budget; frozen pretrained spatial encoder, "
               "higher network input resolution and feature caching replace scratch CNNs. "
               "Retains V3 exploration and V2 task/rewards; success needs physical evaluation." if dino else
               "One visual training seed, small sample budget; changes initial arm noise and "
               "entropy together, retaining V2 rewards and all other untrained network weights. "
               "Success needs physical evaluation." if precision else
               "One visual training seed, small sample budget; changed reward terms, "
               "dense scaling and action penalties together. Success needs physical evaluation."),
    }
    if course:
        plan["suite"] = "visual_course"
        plan["reward_design"] = {
            "step_dt": 0.05, "lift_height_m": 0.04, "continuous_hold_s": 0.5,
            "contact_threshold_n_each_finger": 0.1, "blue_max_speed_m_s": 0.1,
            "reach_distance_scale_m": 0.2, "success_bonus": 200,
            "stages": {
                "lift": {"reach": 2, "lift": 15, "align": 0, "align_fine": 0, "placed": 0},
                "transport": {"reach": 0.5, "lift": 15, "align": 16, "align_fine": 5, "placed": 0},
                "stack": {"reach": 0.5, "lift": 2, "align": 16, "align_fine": 5, "placed": 12},
            },
        }
        plan["caveat"] = ("Visual-only actor, frozen DINO, actual table pickup in every stage. "
                          "Promote only after 64 independent fixed-layout episodes, both seeds >=80% success "
                          "and <=5% workspace failure. Transfers full learned weights, resets optimizer. "
                          "Repeated fixed-layout episodes do not establish generalization. "
                          "Failed stages stop for analysis rather than blindly training later stages.")
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"state": "prepared"})
    return plan


def course_acceptance(reports, criterion):
    return (sum(report["episodes"] for report in reports) == criterion["episodes"]
        and len(reports) == 2 and len({report["seed"] for report in reports}) == 2
        and all(report["frozen_encoder_verified"]
            and report["success_rate"] >= criterion["each_seed_success_rate"]
            and report["workspace_failure_rate"] <= criterion["each_seed_max_workspace_failure_rate"]
            for report in reports))


def prepare(output, epochs=1500, seed=42):
    import torch
    from rlinf.models import get_model

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config, _ = compose_project("franka_lift", config="ablation")
    torch.manual_seed(seed)
    model = get_model(config.actor.model)
    initial = output / "initial_weights.pt"
    torch.save(model.state_dict(), initial)
    conditions, differences = build_conditions(output, epochs, seed)
    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_seed": seed,
        "iterations": epochs,
        "evaluation_seeds": [43, 44],
        "episodes_per_evaluation_seed": 128,
        "initial_weights_sha256": checksum(initial),
        "source_sha256": {name: checksum(ROOT / name) for name in SOURCES},
        "allowed_differences": differences,
        "order": list(CONDITIONS),
        "conditions": conditions,
    }
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"state": "prepared"})
    return plan


def prepare_builtin(output, lift_initial, epochs=1500, seed=42, stack_epochs=1000):
    import torch
    from rlinf.models import get_model

    lift_initial = Path(lift_initial).resolve(strict=True)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(lift_initial, output / "initial_weights.pt")
    config, _ = compose_project("my_robot", config="adaptive")
    torch.manual_seed(seed)
    model = get_model(config.actor.model)
    torch.save(model.state_dict(), output / "stack_initial_weights.pt")
    conditions, differences = build_builtin_conditions(output, epochs, seed, stack_epochs)
    plan = {
        "suite": "builtin_transfer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_seed": seed, "iterations": epochs, "stack_iterations": stack_epochs,
        "evaluation_seeds": [43, 44], "episodes_per_evaluation_seed": 128,
        "initial_weights_sha256": checksum(output / "initial_weights.pt"),
        "extra_initial_weights": {"stack_initial_weights.pt": checksum(output / "stack_initial_weights.pt")},
        "reference_lift_initial": str(lift_initial),
        "source_sha256": {name: checksum(ROOT / name) for name in SOURCES},
        "allowed_differences": differences,
        "order": list(conditions), "conditions": conditions,
        "caveat": "Stack transfer changes several PPO settings together and uses only "
                  "2 environments; it is a recipe screen, not an isolated KL ablation.",
    }
    write_json(output / "plan.json", plan)
    write_json(output / "status.json", {"state": "prepared"})
    return plan


def resume_checkpoint_info(output, plan, checkpoint):
    if len(plan["order"]) != 1:
        raise ValueError("Checkpoint recovery requires a single-condition experiment")
    name = plan["order"][0]
    checkpoint = Path(checkpoint).resolve(strict=True)
    parent = Path(output).resolve() / "runs" / name / name / "checkpoints"
    if checkpoint.parent != parent or not checkpoint.name.startswith("global_step_"):
        raise ValueError("Recovery checkpoint must belong to this experiment")
    step = int(checkpoint.name.removeprefix("global_step_"))
    if not 0 < step < plan["conditions"][name].get("iterations", plan["iterations"]):
        raise ValueError("Recovery checkpoint must precede the final training step")
    actor = checkpoint / "actor"
    weights = actor / "model_state_dict" / "full_weights.pt"
    distributed = actor / "dcp_checkpoint"
    if (not weights.is_file() or not (distributed / ".metadata").is_file()
            or not any(distributed.glob("*.distcp"))):
        raise ValueError("Recovery requires a complete model and optimizer checkpoint")
    return {"checkpoint": str(checkpoint), "from_step": step,
            "checkpoint_sha256": checksum(weights)}


def execute(output, resume_checkpoint=None, plan_file="plan.json"):
    output = Path(output).resolve()
    plan = json.loads((output / plan_file).read_text())
    if checksum(output / "initial_weights.pt") != plan["initial_weights_sha256"]:
        raise ValueError("The frozen initial weights changed")
    for name, expected in plan.get("extra_initial_weights", {}).items():
        if checksum(output / name) != expected:
            raise ValueError(f"The frozen initial weights changed: {name}")
    for name, expected in plan["source_sha256"].items():
        if checksum(ROOT / name) != expected:
            raise ValueError(f"The frozen experiment source changed: {name}")
    for name, expected in plan.get("encoder_source_sha256", {}).items():
        if checksum(name) != expected:
            raise ValueError(f"The pretrained encoder source changed: {name}")
    if plan.get("suite") in ("stack_reward", "stack_precision", "stack_dino", "visual_course"):
        conditions, differences = build_stack_conditions(
            output, plan["iterations"], plan["training_seed"],
            precision=plan["suite"] == "stack_precision",
            dino=plan["suite"] == "stack_dino",
            course=plan["suite"] == "visual_course",
        )
    elif plan.get("suite") == "builtin_transfer":
        conditions, differences = build_builtin_conditions(
            output, plan["iterations"], plan["training_seed"], plan["stack_iterations"]
        )
    else:
        conditions, differences = build_conditions(output, plan["iterations"], plan["training_seed"])
    if conditions != plan["conditions"] or differences != plan["allowed_differences"]:
        raise ValueError("The frozen experiment configurations changed")
    recovery = (resume_checkpoint_info(output, plan, resume_checkpoint)
                if resume_checkpoint is not None else None)
    status = {"state": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "results": {}}
    if recovery is not None:
        status["recovery"] = recovery
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMNI_KIT_ACCEPT_EULA="YES")
    try:
        for name in plan["order"]:
            for source, expected in plan["source_sha256"].items():
                if checksum(ROOT / source) != expected:
                    raise ValueError(f"The frozen experiment source changed: {source}")
            condition = conditions[name]
            project = condition.get("project", "franka_lift")
            iterations = condition.get("iterations", plan["iterations"])
            stack = project == "my_robot"
            status.update(condition=name, phase="training")
            write_json(output / "status.json", status)
            command = ["bash", str(ROOT / "run.sh"), "train", project,
                       "--config", condition.get("recipe", "ablation"), *condition["overrides"]]
            if recovery is not None:
                command.append(f"runner.resume_dir={recovery['checkpoint']}")
            log_name = f"{name}_resume_{recovery['from_step']}.log" if recovery else f"{name}_train.log"
            with (output / log_name).open("w") as log:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                               stderr=subprocess.STDOUT, check=True, timeout=21600 if stack else 7200)
            checkpoint = (output / "runs" / name / name / "checkpoints"
                          / f"global_step_{iterations}" / "actor"
                          / "model_state_dict" / "full_weights.pt")
            checkpoint.resolve(strict=True)
            status.update(phase="evaluation")
            write_json(output / "status.json", status)
            reports = []
            for seed in plan["evaluation_seeds"]:
                destination = output / f"eval_{name}_seed{seed}.json"
                base_command = ([sys.executable, "-m", "tools.stack_evaluation", "--num-envs", "1",
                                 "--config", condition.get("recipe", "adaptive")]
                                if stack else [sys.executable, "-m", "tools.rsl_reference", "evaluate",
                                               "--trainer", "rlinf", "--headless", "--num_envs", "32"])
                if "stage" in condition:
                    base_command.extend(["--stage", condition["stage"]])
                if plan.get("suite") in ("stack_reward", "stack_precision", "stack_dino", "visual_course"):
                    base_command.append("--diagnostics")
                command = [*base_command,
                           "--episodes", str(condition.get("episodes_per_evaluation_seed",
                                                           plan["episodes_per_evaluation_seed"])),
                           "--seed", str(seed), "--checkpoint", str(checkpoint),
                           "--output", str(destination)]
                with (output / f"{name}_eval_seed{seed}.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=1800 if stack else 600)
                reports.append(json.loads(destination.read_text()))
            count = sum(report["episodes"] for report in reports)
            keys = (("success_rate", "workspace_failure_rate", "mean_return") if stack
                    else ("lift_rate", "goal_rate", "held_0_5s_rate", "mean_return"))
            status["results"][name] = {
                "episodes": count,
                "checkpoint_sha256": checksum(checkpoint),
                **{key: sum(report[key] * report["episodes"] for report in reports) / count
                   for key in keys},
            }
            write_json(output / "status.json", status)
            if plan.get("suite") in ("builtin_transfer", "stack_reward", "stack_precision", "stack_dino", "visual_course"):
                status.update(phase="replay")
                write_json(output / "status.json", status)
                replay_base = ([sys.executable, "-m", "tools.stack_evaluation", "--num-envs", "1",
                                "--config", condition.get("recipe", "adaptive")]
                               if stack else [sys.executable, "-m", "tools.rsl_reference", "evaluate",
                                              "--trainer", "rlinf", "--headless", "--num_envs", "1"])
                if "stage" in condition:
                    replay_base.extend(["--stage", condition["stage"]])
                if plan.get("suite") in ("stack_reward", "stack_precision", "stack_dino", "visual_course"):
                    replay_base.append("--diagnostics")
                command = [*replay_base, "--episodes", "2", "--seed", "43",
                           "--checkpoint", str(checkpoint), "--output", str(output / f"replay_{name}.json"),
                           "--video", str(output / f"replay_{name}.mp4")]
                with (output / f"{name}_replay.log").open("w") as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=True, timeout=600)
                status["results"][name]["replay_video"] = f"replay_{name}.mp4"
                write_json(output / "status.json", status)
            if "acceptance" in condition:
                accepted = course_acceptance(reports, condition["acceptance"])
                status["results"][name]["accepted"] = accepted
                write_json(output / "status.json", status)
                if not accepted:
                    status.update(state="acceptance_failed", phase="awaiting_analysis",
                                  finished_at=datetime.now(timezone.utc).isoformat(),
                                  reason="Independent physical acceptance failed; later stages were not started")
                    write_json(output / "status.json", status)
                    return
        comparison = {}
        if "adaptive" in status["results"]:
            comparison["goal_rate_difference"] = (status["results"]["adaptive"]["goal_rate"]
                                                  - status["results"]["fixed"]["goal_rate"])
        status.update(state="completed", phase="completed",
                      **comparison,
                      finished_at=datetime.now(timezone.utc).isoformat(),
                      caveat=plan.get("caveat", "One paired training seed; this isolates the learning-rate "
                                     "rule, not every remaining difference from RSL-RL."))
        write_json(output / "status.json", status)
    except Exception as error:
        status.update(state="failed", error=str(error))
        write_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", choices=("kl_pair", "builtin_transfer", "stack_reward", "stack_precision", "stack_dino", "visual_course"), default="kl_pair")
    parser.add_argument("--stack-epochs", type=int, default=1000)
    parser.add_argument("--lift-initial-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--plan-file", default="plan.json")
    parser.add_argument("--execute", action="store_true")
    options = parser.parse_args()
    if options.epochs < 1 or options.stack_epochs < 1:
        parser.error("epochs must be positive")
    if options.execute:
        execute(options.output, options.resume_checkpoint, options.plan_file)
    else:
        if options.resume_checkpoint is not None:
            parser.error("--resume-checkpoint requires --execute")
        if options.suite in ("stack_reward", "stack_precision", "stack_dino", "visual_course"):
            if options.initial_checkpoint is None:
                parser.error(f"{options.suite} requires --initial-checkpoint")
            plan = prepare_stack(options.output, options.initial_checkpoint, options.epochs,
                                 options.seed, precision=options.suite == "stack_precision",
                                 dino=options.suite == "stack_dino", course=options.suite == "visual_course")
        elif options.suite == "builtin_transfer":
            if options.lift_initial_checkpoint is None:
                parser.error("builtin_transfer requires --lift-initial-checkpoint")
            plan = prepare_builtin(options.output, options.lift_initial_checkpoint,
                                   options.epochs, options.seed, options.stack_epochs)
        else:
            plan = prepare(options.output, options.epochs, options.seed)
        print(json.dumps({"output": str(options.output.resolve()),
                          "initial_weights_sha256": plan["initial_weights_sha256"],
                          "allowed_differences": plan["allowed_differences"]}, indent=2))


if __name__ == "__main__":
    main()