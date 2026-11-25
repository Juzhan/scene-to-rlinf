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
