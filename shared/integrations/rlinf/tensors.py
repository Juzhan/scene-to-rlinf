"""Construction and tensor validation for the native RLinf entry point."""

import torch


def _make_environment(parameters, num_envs, seed, horizon, *, factory=None):
    if not callable(factory):
        raise ValueError("Register a callable environment factory before training")
    environment, owner = factory(parameters, num_envs, seed, horizon)
    try:
        for name in ("reset", "step", "close"):
            if not callable(getattr(environment, name, None)):
                raise TypeError(f"Environment must provide {name}()")
        if not hasattr(environment, "device"):
            raise TypeError("Environment must expose its device")
        torch.device(environment.device)
        action_dim = getattr(environment, "action_dim", None)
        if type(action_dim) is not int or action_dim < 1:
            raise ValueError("Environment action_dim must be a positive integer")
        if action_dim != parameters.action_dim:
            raise ValueError(f"Configured action_dim={parameters.action_dim}, actual={action_dim}")
        if owner is not None and not callable(getattr(owner, "close", None)):
            raise TypeError("Resource owner must provide close() or be None")
        if getattr(environment, "num_envs", None) != num_envs:
            raise ValueError("Factory must preserve num_envs")
    except BaseException:
        try:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        finally:
            close = getattr(owner, "close", None)
            if callable(close):
                close()
        raise
    return environment, owner


def _validate_config(cfg, *, factory=None):
    if not cfg.auto_reset or cfg.ignore_terminations or cfg.use_rel_reward:
        raise ValueError("Assembly training requires auto_reset=True and unchanged task rewards/terminations")
    if cfg.video_cfg.save_video or cfg.get("enable_offload", False):
        raise ValueError("Assembly recording and offload are not implemented")
    if type(cfg.max_episode_steps) is not int or cfg.max_episode_steps < 1:
        raise ValueError("max_episode_steps must be a positive integer")
    if type(cfg.init_params.action_dim) is not int or cfg.init_params.action_dim < 1:
        raise ValueError("action_dim must be a positive integer")
    if not callable(factory):
        raise ValueError("Register a callable environment factory before training")
    if "process_isolation" in cfg or "process_timeout_s" in cfg:
        raise ValueError("Process management belongs to the backend integration, not task configuration")


def _wrap_observations(observations, num_envs):
    if not isinstance(observations, dict) or not observations:
        raise ValueError("Training requires a nonempty observation dictionary")
    result = {}
    for name, value in observations.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != num_envs:
            raise ValueError(f"Observation {name} must be a tensor with leading dimension num_envs")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Observation {name} must be finite")
        result[name] = value.detach().clone()
    return result


def _prepare_actions(actions, expected, device):
    try:
        actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"Assembly training requires finite actions of shape {expected}") from error
    if tuple(actions.shape) != expected or not torch.isfinite(actions).all():
        raise ValueError(f"Assembly training requires finite actions of shape {expected}")
    return actions
