"""Optional episode-local accumulators for Python tasks; reset by the native adapter."""
import torch


class TaskMemory:
    def __init__(self):
        self._values = {}

    def _get(self, name, value, shape=None):
        shape = value.shape if shape is None else shape
        if name not in self._values:
            self._values[name] = torch.zeros(shape, device=value.device, dtype=value.dtype)
        result = self._values[name]
        if tuple(result.shape) != tuple(shape):
            raise ValueError(f'Task memory shape changed for {name}')
        return result

    def hold(self, name, condition, dt):
        elapsed = self._get(name, dt)
        elapsed.copy_(torch.where(condition, elapsed + dt, 0))
        return elapsed.clone()

    def mean(self, name, value, window=None):
        count = self._get(name + '/count', value, (value.shape[0],))
        if window is None:
            total = self._get(name + '/sum', value)
            total.add_(value)
            count.add_(1)
            return total / count.reshape(-1, *([1] * (value.ndim - 1)))
        history = self._get(name + '/history', value, (value.shape[0], window, *value.shape[1:]))
        rows = torch.arange(value.shape[0], device=value.device)
        history[rows, count.long() % window] = value
        count.add_(1)
        return history.sum(dim=1) / count.clamp(max=window).reshape(-1, *([1] * (value.ndim - 1)))

    def reset(self, env_ids=None):
        for value in self._values.values():
            value[slice(None) if env_ids is None else env_ids] = 0
