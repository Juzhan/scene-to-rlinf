"""Torch policy trained by backpropagating through the Newton rollout."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .rollout import OBS_DIM

MAX_SPEED = 0.20  # m/s; larger commands drive the cubes into deep contact and unstable gradients


class VelocityPolicy(nn.Module):
    """Maps the placement observation to a world-frame velocity and a grip level."""

    def __init__(self, hidden: int = 128, obs_dim: int = OBS_DIM, max_speed: float = MAX_SPEED):
        self.max_speed = max_speed
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        # Offsets are centimetre-scale; scale them to O(1) before the tanh layers.
        raw = self.body(torch.cat((10.0 * obs[:, :-1], obs[:, -1:]), -1))
        velocity = self.max_speed * torch.tanh(raw[:, :3])
        grip = torch.sigmoid(raw[:, 3:4])
        return torch.cat((velocity, grip), -1), raw
