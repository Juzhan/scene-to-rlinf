"""Check the closed-loop BPTT gradient against finite differences.

Run before trusting a training run: ``python -m projects.box_stack_diffsim.diffsim.validate``.
A random policy is rolled out; the analytic directional derivative of the
episode loss along a random parameter direction is compared with a central
difference.  ``--case contact`` starts the held cube resting on the base.
"""
from __future__ import annotations

import argparse
import json

import torch
import warp as wp

from .policy import VelocityPolicy
from .rollout import make_sim
from .scene import read_scene
from .train import episode_loss

CASES = {"free": (0.0, 0.0, 0.10), "contact": (0.13, -0.18, 0.062)}


def check(case: str, horizon: int, worlds: int, epsilon: float, seed: int) -> dict:
    torch.manual_seed(seed)
    sim = make_sim(read_scene(), worlds=worlds)
    policy = VelocityPolicy(64).to(sim.device)
    for p in policy.parameters():
        torch.nn.init.normal_(p, std=0.3)
    offsets = torch.tensor(CASES[case]).repeat(worlds, 1)
    offsets[:, :2] += 0.01 * torch.randn(worlds, 2)

    loss, _ = episode_loss(sim, policy, offsets, horizon)
    params = list(policy.parameters())
    grads = torch.autograd.grad(loss, params)
    direction = [torch.randn_like(p) for p in params]
    analytic = float(sum((g * d).sum() for g, d in zip(grads, direction)))

    def shifted(sign: float) -> float:
        with torch.no_grad():
            for p, d in zip(params, direction):
                p.add_(sign * epsilon * d)
            value = float(episode_loss(sim, policy, offsets, horizon)[0])
            for p, d in zip(params, direction):
                p.sub_(sign * epsilon * d)
        return value

    reference = (shifted(1.0) - shifted(-1.0)) / (2.0 * epsilon)
    relative = abs(analytic - reference) / max(1e-12, abs(reference))
    return {"case": case, "horizon": horizon, "worlds": worlds, "loss": float(loss),
            "analytic": analytic, "finite_difference": reference, "relative_error": relative,
            "nonfinite_steps": sim.nonfinite, "passed": bool(relative < 0.05)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="free", choices=sorted(CASES))
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--worlds", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    wp.config.log_level = wp.LOG_WARNING
    report = check(args.case, args.horizon, args.worlds, args.epsilon, args.seed)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
