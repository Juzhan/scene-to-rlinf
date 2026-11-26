"""What policy.py builds networks with: ``mlp`` for a plain network and ``flat`` to join inputs.

A layer's ``forward(obs)`` returns a Tensor, or a dict of Tensors when its result has parts the next layer reads by
name::

    def forward(self, obs: UpperObs) -> dict:
        return {'goal': goal,        # [B, 3] 目标位置（m，环境坐标）
                'heading': heading}  # [B, 1] 目标朝向（rad）

and the next layer reads ``obs.upper.goal``. The keys label the editor's arrow between the layers, the comment after
each entry (shape and meaning) is their tooltip; observation.py reads both from the source, and sizes a comment does
not give are measured on the first step. The trained layer returns its actions as one Tensor.
"""
import torch
from torch import nn


def mlp(*sizes, activation=nn.ELU):
    """``mlp(256, 128, 12)``: hidden layers of 256 and 128 units with ``activation`` between them and 12 outputs.
    The input size is taken from the first batch (LazyLinear), so concatenate whatever the layer reads."""
    if len(sizes) < 1:
        raise ValueError('mlp needs at least the output size')
    layers = []
    for size in sizes[:-1]:
        layers += [nn.LazyLinear(size), activation()]
    return nn.Sequential(*layers, nn.LazyLinear(sizes[-1]))


def flat(*tensors):
    """Concatenate [B, ...] tensors into [B, features] (each flattened after the batch axis)."""
    return torch.cat([tensor.flatten(1) for tensor in tensors], -1)

def inverse_rotate(vector, quaternion):
    """Rotate ``vector`` into the frame of a wxyz ``quaternion`` (broadcasting over the batch axis)."""
    xyz = quaternion[..., 1:]
    cross = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector - quaternion[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def yaw(quaternion):
    """Heading about +Z in radians, shape (..., ); wxyz quaternion."""
    w, x, y, z = quaternion.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def relative_goal(position, quaternion, goal_position, goal_heading):
    """Goal position and heading error in the robot frame: ``(relative [..., 3], error [..., 1])``."""
    heading = yaw(quaternion)
    offset = goal_position - position
    c, s = heading.cos(), heading.sin()
    relative = torch.stack((c * offset[..., 0] + s * offset[..., 1],
                            -s * offset[..., 0] + c * offset[..., 1], offset[..., 2]), -1)
    difference = goal_heading - heading
    wrapped = (difference + torch.pi) % (2 * torch.pi)
    error = torch.where((wrapped == 0) & (difference > 0), torch.pi, wrapped - torch.pi)
    return relative, error.unsqueeze(-1)
