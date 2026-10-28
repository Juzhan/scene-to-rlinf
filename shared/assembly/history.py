"""Per-environment observation windows shared by online components and offline data.

One window holds the last ``length`` decision-time frames of one observation, ordered
oldest -> newest along axis 1 (``[B, T, *shape]``). Semantics follow the native Isaac Lab /
mjlab ``CircularBuffer`` so a project can reason about either: after a reset the first pushed
frame fills every slot (repeat-first padding) and ``valid`` marks how many of them were really
observed in the current episode. Windows never span environments, resets or missing steps.
"""
import torch


def validate_history(history, observations, role='component'):
    """``{observation: frames}`` for selected observations; frames is an int >= 2."""
    if history is None:
        return {}
    if not isinstance(history, dict):
        raise ValueError(f'{role}.history must map selected observations to a frame count')
    for source, length in history.items():
        if source not in observations:
            raise ValueError(f'{role}.history: {source!r} is not one of the selected observations')
        if type(length) is not int or length < 2:
            raise ValueError(f'{role}.history.{source} must be an integer >= 2 (omit it for a single frame)')
    return dict(history)


class HistoryWindow:
    """Fixed-length window per environment; pushes and resets touch only the given rows."""

    def __init__(self, example, length):
        if type(length) is not int or length < 2:
            raise ValueError('History length must be an integer >= 2')
        self.length = length
        self.buffer = torch.zeros((example.shape[0], length, *example.shape[1:]),
                                  dtype=example.dtype, device=example.device)
        self.count = torch.zeros(example.shape[0], dtype=torch.long, device=example.device)

    def _rows(self, rows):
        if rows is None or rows is True:
            return torch.ones_like(self.count, dtype=torch.bool)
        if isinstance(rows, torch.Tensor) and rows.dtype == torch.bool:
            return rows.to(self.count.device)
        mask = torch.zeros_like(self.count, dtype=torch.bool)
        mask[rows] = True
        return mask

    def _expand(self, mask):
        return mask.view(-1, *([1] * (self.buffer.dim() - 1)))

    def push(self, frame, rows=None):
        """Append ``frame`` [B, ...] as the newest entry of the selected rows (no host sync)."""
        rows = self._rows(rows)
        frame = frame.to(self.buffer.dtype)
        repeated = frame.unsqueeze(1).expand_as(self.buffer)
        shifted = torch.cat((self.buffer[:, 1:], frame.unsqueeze(1)), dim=1)
        fresh = rows & (self.count == 0)
        value = torch.where(self._expand(rows), shifted, self.buffer)
        value = torch.where(self._expand(fresh), repeated, value)
        self.buffer.copy_(value)
        self.count.add_(rows.long()).clamp_(max=self.length)
        return self.buffer

    def peek(self, frame):
        """Show an empty window as ``frame`` repeated without counting it as observed."""
        empty = self._expand(self.count == 0)
        self.buffer.copy_(torch.where(empty, frame.to(self.buffer.dtype).unsqueeze(1).expand_as(self.buffer), self.buffer))
        return self.buffer

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.buffer[ids] = 0
        self.count[ids] = 0

    @property
    def valid(self):
        """[B, T] bool: True where the slot holds a frame observed in the current episode."""
        steps = torch.arange(self.length, device=self.count.device)
        return steps.unsqueeze(0) >= (self.length - self.count).unsqueeze(1)


def history_windows(frames, length, *, episode=None, step=None):
    """Offline windows with the online construction: ``frames`` [N, ...] in time order.

    Frames must be recorded at the component's decision rate (one per decision; nothing is
    resampled). A new window starts when ``episode`` changes or ``step`` is not the previous
    step + 1 (a missing step). Returns ``(windows [N, T, ...], valid [N, T])``.
    """
    frames = torch.as_tensor(frames)
    count = frames.shape[0]
    episode = None if episode is None else torch.as_tensor(episode).reshape(-1)
    step = None if step is None else torch.as_tensor(step).reshape(-1)
    for name, value in (('episode', episode), ('step', step)):
        if value is not None and value.shape[0] != count:
            raise ValueError(f'{name} must have one entry per frame')
    window = HistoryWindow(frames[:1], length)
    windows = frames.new_zeros((count, length, *frames.shape[1:]))
    valid = torch.zeros((count, length), dtype=torch.bool)
    for index in range(count):
        boundary = index == 0
        if not boundary and episode is not None and episode[index] != episode[index - 1]:
            boundary = True
        if not boundary and step is not None and step[index] != step[index - 1] + 1:
            boundary = True
        if boundary:
            window.reset()
        window.push(frames[index:index + 1])
        windows[index] = window.buffer[0]
        valid[index] = window.valid[0]
    return windows, valid
