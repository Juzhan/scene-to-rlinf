"""Explicit small simulator/policy check; never updates model parameters.

An object project (Python Task + policy.py) builds its environment through the shared
registry and calls the project model directly, so this runs with the simulator group
alone. Legacy native-term projects keep the training framework's own env/model path.
"""

from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path

import torch


def report_path(cfg):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return Path(cfg.runner.logger.log_path) / "checks" / f"{timestamp}.json"


def write_report(cfg, report):
    destination = report_path(cfg)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")
    return report


def project_policy(cfg, module):
    """The trained component's model, untrained or from the weights scene.yaml names."""
    project = module.project
    return project.model_builder()(cfg.actor.model, torch.float32)


def smoke_project(cfg, module, steps=16):
    """Object projects: reset, forward, step; nobody else owns the policy, so no trainer is needed."""
    from shared.assembly.project import environment_factory

    parameters = cfg.env.train.init_params
    factory = environment_factory(parameters.id)
    if factory is None:
        raise ValueError(f"Project {parameters.id} is not registered: run check first")
    schema = import_module(module.__name__ + ".observation")
    count = min(4, cfg.env.train.total_num_envs)
    model = project_policy(cfg, module).to(parameters.get("device", "cuda:0")).eval()
    environment, _ = factory(parameters, count, cfg.env.train.seed, cfg.env.train.max_episode_steps)
    reset_count = 0
    try:
        with torch.inference_mode():
            observations, _ = environment.reset()
            for _ in range(steps):
                actions = model(schema.Obs({key: value.to(environment.device) if torch.is_tensor(value) else value
                                            for key, value in observations.items()}))
                if not torch.isfinite(actions).all():
                    raise ValueError("Non-finite policy actions")
                observations, rewards, terminated, truncated, _ = environment.step(actions)
                if not torch.isfinite(rewards).all():
                    raise ValueError("Non-finite simulator transition")
                reset_count += int((terminated | truncated).sum())
            report = {"task": parameters.id, "num_envs": count, "steps": steps,
                      "device": str(environment.device), "action_shape": list(actions.shape),
                      "resets": reset_count, "contract_passed": True, "task_success_evaluated": False}
    finally:
        environment.close()
    report["clean_exit"] = True
    return write_report(cfg, report)


def smoke(cfg, module=None):
    try:
        from rlinf.envs import get_env_cls
        from rlinf.models import get_model
    except ModuleNotFoundError:
        if module is None or not (Path(module.__file__).parent / "task.py").is_file():
            raise ValueError("This project needs the training framework: tools/setup_env.py --with train") from None
        return smoke_project(cfg, module)

    model = get_model(cfg.actor.model).eval()
    count = min(4, cfg.env.train.total_num_envs)
    environment = get_env_cls(cfg.env.train.env_type, cfg.env.train)(cfg.env.train, count, 0, 1, None)
    reset_count = 0
    try:
        model.to(environment.device)
        with torch.inference_mode():
            observations, _ = environment.reset()
            for _ in range(16):
                actions, _ = model.predict_action_batch(observations, mode="eval")
                obs_list, rewards, terminated, truncated, infos = environment.chunk_step(actions)
                observations = obs_list[-1]
                if not torch.isfinite(rewards).all() or not torch.isfinite(actions).all():
                    raise ValueError("Non-finite simulator transition")
                dones = terminated | truncated
                if dones.any() and "final_observation" not in infos[-1]:
                    raise ValueError("Auto-reset lost the terminal observation")
                reset_count += int(dones.sum())
            report = {"task": cfg.env.train.init_params.id, "num_envs": count, "steps": 16,
                      "device": str(environment.device), "action_shape": list(actions.shape),
                      "resets": reset_count, "contract_passed": True, "task_success_evaluated": False}
    finally:
        environment.close()
    report["clean_exit"] = True
    return write_report(cfg, report)
