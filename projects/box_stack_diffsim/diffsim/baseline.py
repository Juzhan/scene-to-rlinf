"""Scripted controller on the same differentiable simulation.

Answers "is the task reachable with this action space and horizon?" before any
learning: a proportional velocity command to the base top, released on arrival.
``python -m projects.box_stack_diffsim.diffsim.baseline --horizon 60``
"""
from __future__ import annotations

import argparse
import json

import torch
import warp as wp

from .policy import MAX_SPEED
from .rollout import evaluate, make_sim, start_offsets
from .scene import read_scene


def scripted(gain: float, max_speed: float, release: float, hover: float = 0.02):
    def act(obs: torch.Tensor) -> torch.Tensor:
        error = obs[:, :3]
        # Stay above the base top until aligned in x/y; a side hit launches the base.
        aligned = error[:, :2].norm(dim=-1, keepdim=True) < 0.005
        lift = torch.zeros_like(error)
        lift[:, 2:] = hover * (~aligned).float()
        velocity = (-gain * (error - lift)).clamp(-max_speed, max_speed)
        grip = (error.norm(dim=-1, keepdim=True) > release).float()
        return torch.cat((velocity, grip), -1)
    return act


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--starts", type=int, default=64)
    parser.add_argument("--spread", type=float, default=0.03)
    parser.add_argument("--gain", type=float, default=6.0)
    parser.add_argument("--max-speed", type=float, default=MAX_SPEED)
    parser.add_argument("--release", type=float, default=0.006)
    args = parser.parse_args(argv)

    wp.config.log_level = wp.LOG_WARNING
    sim = make_sim(read_scene(), worlds=args.starts)
    offsets = start_offsets(args.starts, args.spread, torch.Generator().manual_seed(1234))
    report = evaluate(sim, scripted(args.gain, args.max_speed, args.release), offsets, args.horizon)
    print(json.dumps({**vars(args), **report}, indent=2))


if __name__ == "__main__":
    main()
