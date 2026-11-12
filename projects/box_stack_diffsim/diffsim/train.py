"""Full BPTT for the box-stacking scene on Newton differentiable physics.

Each decision step is a torch op backed by a Warp tape (see ``rollout.py``), so
the loss gradient reaches the policy through the actions *and* through every
later observation: the SuperMimic recipe with Newton in place of a learned
world model.  ``--truncate K`` cuts the gradient every ``K`` steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import warp as wp

from .policy import VelocityPolicy
from .rollout import BASE, HELD, evaluate, make_sim, observe, placement, stacked, start_offsets, step
from .scene import read_scene

RUNS = Path(__file__).resolve().parents[1] / "runs"


def episode_loss(sim, policy, held_offset, horizon: int, truncate: int = 0,
                 running_weight: float = 0.1) -> tuple[torch.Tensor, dict]:
    q, qd = sim.state0(held_offset)
    running = torch.zeros(sim.worlds, device=sim.device)
    for k in range(horizon):
        if truncate and k and k % truncate == 0:
            q, qd = q.detach(), qd.detach()
        action, _ = policy(observe(sim, q, qd))
        q, qd = step(sim, q, qd, action, k)
        error, shift = placement(sim, q)
        running = running + error.square() + 10.0 * shift.square()
    error, shift = placement(sim, q)
    speed = qd[sim.held, :3].norm(dim=-1)
    loss = error.square() + 10.0 * shift.square() + 0.05 * speed.square() + running_weight * running / horizon
    return loss.mean(), {"final_error_m": float(error.detach().mean()), "base_shift_m": float(shift.detach().mean()),
                         "success": stacked(sim, q.detach(), qd.detach())}


class StartPool:
    """Success Guided Sampling over a ladder of starts, from the full task to just above the stack.

    Start ``i`` is ``far_i + s_i (near_i - far_i)``; configurations are drawn with weight
    ``p^t (1-p)^(1-t)`` on their running success rate ``p``, mixed with uniform.
    """

    def __init__(self, sim, size: int, spread: float, generator: torch.Generator,
                 target: float = 0.5, uniform: float = 0.2):
        scene = sim.scene
        stack = torch.tensor(scene.world_positions[BASE]) + torch.tensor([0.0, 0.0, sim.gap])
        above = stack - torch.tensor(scene.world_positions[HELD])
        far = start_offsets(size, spread, generator)
        near = above + torch.cat((0.01 * (2 * torch.rand(size, 2, generator=generator) - 1),
                                  0.005 + 0.035 * torch.rand(size, 1, generator=generator)), -1)
        self.level = torch.rand(size, 1, generator=generator)
        self.offsets = far + self.level * (near - far)
        self.success = torch.zeros(size)
        self.target, self.uniform = target, uniform

    def sample(self, count: int, generator: torch.Generator) -> torch.Tensor:
        p = self.success
        weight = p.pow(self.target) * (1 - p).pow(1 - self.target)
        weight = weight / weight.sum() if weight.sum() > 0 else torch.full_like(p, 1 / len(p))
        weight = (1 - self.uniform) * weight + self.uniform / len(p)
        return torch.multinomial(weight, count, replacement=True, generator=generator)

    def update(self, index: torch.Tensor, success: torch.Tensor) -> None:
        self.success[index] = 0.7 * self.success[index] + 0.3 * success.float().cpu()


def train(args) -> dict:
    wp.config.log_level = wp.LOG_WARNING
    torch.manual_seed(args.seed)
    scene = read_scene()
    sim = make_sim(scene, worlds=args.worlds, device=args.device)
    eval_sim = make_sim(scene, worlds=args.eval_starts, device=args.device)
    eval_offsets = start_offsets(args.eval_starts, args.spread, torch.Generator().manual_seed(1234))
    generator = torch.Generator().manual_seed(args.seed)
    pool = StartPool(sim, args.sgs_pool, args.spread, generator) if args.sgs_pool else None
    policy = VelocityPolicy(args.hidden).to(args.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    run_dir = RUNS / time.strftime("bptt_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    act = lambda obs: policy(obs)[0]

    history, evaluations = [], []
    best = {"success_rate": -1.0, "mean_error_m": float("inf")}
    started = time.time()
    for iteration in range(args.iterations + 1):
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            report = {"iteration": iteration, **evaluate(eval_sim, act, eval_offsets, args.horizon),
                      "nonfinite_steps": sim.nonfinite, "seconds": round(time.time() - started, 1)}
            evaluations.append(report)
            print(json.dumps(report), flush=True)
            if (report["success_rate"], -report["mean_error_m"]) > (best["success_rate"], -best["mean_error_m"]):
                best = report
                torch.save(policy.state_dict(), run_dir / "best.pt")
        if iteration == args.iterations:
            break
        if pool is None:
            offsets = start_offsets(args.worlds, args.spread, generator)
        else:
            index = pool.sample(args.worlds, generator)
            offsets = pool.offsets[index]
        loss, info = episode_loss(sim, policy, offsets, args.horizon, args.truncate, args.running_weight)
        success = info.pop("success")
        info["train_success"] = float(success.float().mean())
        if pool is not None:
            pool.update(index, success)
            info["pool_level"] = float(pool.level[index].mean())
            info["pool_success"] = float(pool.success.mean())
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
        if torch.isfinite(norm):
            optimizer.step()
        history.append({"loss": float(loss), "grad_norm": float(norm), **info})
        if iteration % args.log_every == 0:
            print(f"iter {iteration:4d} loss {float(loss):.5f} grad {float(norm):.3g} "
                  f"err {info['final_error_m']:.4f} shift {info['base_shift_m']:.4f} "
                  f"succ {info['train_success']:.2f} level {info.get('pool_level', 0.0):.2f} "
                  f"nonfinite {sim.nonfinite} ({time.time() - started:.0f}s)", flush=True)

    torch.save(policy.state_dict(), run_dir / "policy.pt")
    summary = {"args": vars(args), "best": best, "evaluations": evaluations, "history": history,
               "nonfinite_steps": sim.nonfinite, "wall_seconds": round(time.time() - started, 1)}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"run_dir": str(run_dir), "best": best}, indent=2))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=90, help="60 Hz decision steps per episode")
    parser.add_argument("--worlds", type=int, default=32, help="parallel episodes per update")
    parser.add_argument("--truncate", type=int, default=0)
    parser.add_argument("--running-weight", type=float, default=0.1,
                        help="weight of the per-step error; it pulls the cube down to stack height early")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--sgs-pool", type=int, default=0,
                        help="size of the Success Guided Sampling start pool; 0 trains on the full task only")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--spread", type=float, default=0.03, help="x/y start offset radius in metres")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-starts", type=int, default=64)
    return train(parser.parse_args(argv))


if __name__ == "__main__":
    main()
