"""Validation of the observation dictionary a native environment returns.

The batch axis comes first and every value is a finite tensor; the training
transport and preview share this check.
"""


def checked_observations(observations, num_envs, *, label="Observation"):
    """Detached copies of each tensor, refusing anything but finite [num_envs, ...] tensors."""
    import torch  # imported here so option parsing and --help work without the simulator group

    if not isinstance(observations, dict) or not observations:
        raise ValueError(f"{label}s must be a nonempty dictionary")
    result = {}
    for name, value in observations.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != num_envs:
            raise ValueError(f"{label} {name} must be a tensor with leading dimension num_envs")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{label} {name} must be finite")
        result[name] = value.detach().clone()
    return result
