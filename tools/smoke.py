"""Explicit small simulator/policy check; never updates model parameters."""

from datetime import datetime, timezone
import json
from pathlib import Path

import torch


def smoke(cfg):
    from rlinf.envs import get_env_cls
    from rlinf.models import get_model

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
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    destination = Path(cfg.runner.logger.log_path) / "checks" / f"{timestamp}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Report: {destination}")
    return report