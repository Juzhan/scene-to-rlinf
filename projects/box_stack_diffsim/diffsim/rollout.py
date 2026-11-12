"""Differentiable Newton step exposed to torch autograd.

One decision step is ``substeps`` Newton steps recorded on its own ``wp.Tape``.
:class:`_Step` turns it into a torch op ``(q, qd, action) -> (q', qd')`` whose
backward runs that tape, so a rollout written in torch backpropagates through
the dynamics *and* the policy's dependence on the observed state (full BPTT).

Action per world is ``(vx, vy, vz, grip)`` in world coordinates.  The held
cube's linear velocity is blended toward the command by ``grip``; ``grip = 0``
leaves it to gravity and contacts, so the policy can release the cube.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import warp as wp

import newton

from .scene import SceneBodies, builder, model

# 60 Hz decisions; 1/240 s substeps make two stacked dynamic cubes unstable.
SIM_DT = 1.0 / 960.0
DEFAULT_SUBSTEPS = 16
HELD = "cube_upper"
BASE = "cube_lower"
OBS_DIM = 10


@dataclass
class Sim:
    model: newton.Model
    scene: SceneBodies
    solver: object
    pipeline: object
    control: object
    worlds: int
    dt: float
    substeps: int
    device: str
    held: torch.Tensor
    base: torch.Tensor
    held_world: wp.array
    contacts: list = field(default_factory=list)
    nonfinite: int = 0

    def body(self, name: str, world: int = 0) -> int:
        return self.scene.index(name, world)

    @property
    def gap(self) -> float:
        return self.scene.sizes[HELD][2]

    def base_start(self) -> torch.Tensor:
        return torch.tensor(self.scene.world_positions[BASE], device=self.device)

    def contacts_at(self, index: int):
        # Each substep keeps its own buffer: the tape reads it again in backward.
        while len(self.contacts) <= index:
            self.contacts.append(self.pipeline.contacts())
        return self.contacts[index]

    def state0(self, held_offset: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-velocity state; ``held_offset[w]`` shifts the held cube from its scene pose."""
        q = wp.to_torch(self.model.body_q).detach().clone()
        start = torch.tensor(self.scene.world_positions[HELD], device=self.device)
        q[self.held, :3] = start + held_offset.to(self.device)
        return q, torch.zeros(q.shape[0], 6, device=self.device)


def make_sim(scene: SceneBodies, *, worlds: int = 1, dt: float = SIM_DT,
             substeps: int = DEFAULT_SUBSTEPS, device: str = "cuda:0") -> Sim:
    # Worlds overlap in space; Newton only collides shapes of the same world.
    built = model(builder(scene), worlds=worlds, spacing=0.0)
    held_world = np.full(scene.body_count * worlds, -1, dtype=np.int32)
    for w in range(worlds):
        held_world[scene.index(HELD, w)] = w
    index = lambda name: torch.tensor([scene.index(name, w) for w in range(worlds)], device=device)
    return Sim(built, scene, newton.solvers.SolverSemiImplicit(built),
               newton.CollisionPipeline(built, requires_grad=True), built.control(),
               worlds, dt, substeps, device, index(HELD), index(BASE),
               wp.array(held_world, dtype=int, device=device))


@wp.kernel
def _inject(qd_in: wp.array(dtype=wp.spatial_vector), held_world: wp.array(dtype=int),
            actions: wp.array(dtype=wp.vec4), qd_out: wp.array(dtype=wp.spatial_vector)):
    body = wp.tid()
    current = qd_in[body]
    world = held_world[body]
    if world >= 0:
        action = actions[world]
        blend = wp.clamp(action[3], 0.0, 1.0)
        target = wp.spatial_vector(action[0], action[1], action[2], 0.0, 0.0, 0.0)
        current = current + blend * (target - current)
    qd_out[body] = current


def _simulate(sim: Sim, q: wp.array, qd: wp.array, act: wp.array, step: int, requires_grad: bool):
    state = sim.model.state(requires_grad=requires_grad)
    state.body_q, state.body_qd = q, qd
    for s in range(sim.substeps):
        velocity = wp.empty_like(state.body_qd, requires_grad=requires_grad)
        wp.launch(_inject, dim=velocity.shape[0], inputs=[state.body_qd, sim.held_world, act],
                  outputs=[velocity])
        state.body_qd = velocity
        contacts = sim.contacts_at(step * sim.substeps + s)
        sim.pipeline.collide(state, contacts)
        nxt = sim.model.state(requires_grad=requires_grad)
        sim.solver.step(state, nxt, sim.control, contacts, sim.dt)
        state = nxt
    return state


class _Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, qd, action, sim, step):
        q_in = wp.from_torch(q.detach().contiguous().clone(), dtype=wp.transform, requires_grad=True)
        qd_in = wp.from_torch(qd.detach().contiguous().clone(), dtype=wp.spatial_vector, requires_grad=True)
        act = wp.from_torch(action.detach().float().contiguous().clone(), dtype=wp.vec4, requires_grad=True)
        tape = wp.Tape()
        with tape:
            final = _simulate(sim, q_in, qd_in, act, step, True)
        ctx.saved = (sim, tape, q_in, qd_in, act, final)
        return wp.to_torch(final.body_q).clone(), wp.to_torch(final.body_qd).clone()

    @staticmethod
    def backward(ctx, grad_q, grad_qd):
        sim, tape, q_in, qd_in, act, final = ctx.saved
        ctx.saved = None
        tape.backward(grads={
            final.body_q: wp.from_torch(grad_q.contiguous().clone(), dtype=wp.transform),
            final.body_qd: wp.from_torch(grad_qd.contiguous().clone(), dtype=wp.spatial_vector)})
        grads = [wp.to_torch(a.grad).clone() for a in (q_in, qd_in, act)]
        tape.zero()
        # The experimental contact backward can blow up when a contact appears inside the step.
        if not all(torch.isfinite(g).all() for g in grads):
            sim.nonfinite += 1
            grads = [torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0) for g in grads]
        return grads[0], grads[1], grads[2], None, None


def step(sim: Sim, q: torch.Tensor, qd: torch.Tensor, action: torch.Tensor, k: int):
    """Advance one decision step; differentiable when torch grad mode is on."""
    if torch.is_grad_enabled():
        return _Step.apply(q, qd, action, sim, k)
    final = _simulate(sim, wp.from_torch(q.contiguous().clone(), dtype=wp.transform),
                      wp.from_torch(qd.contiguous().clone(), dtype=wp.spatial_vector),
                      wp.from_torch(action.float().contiguous().clone(), dtype=wp.vec4), k, False)
    return wp.to_torch(final.body_q).clone(), wp.to_torch(final.body_qd).clone()


def observe(sim: Sim, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """Held offset to the base top, held velocity, base displacement, bias."""
    held, base = q[sim.held, :3], q[sim.base, :3]
    stack = base + torch.tensor([0.0, 0.0, sim.gap], device=q.device)
    return torch.cat((held - stack, qd[sim.held, :3], base - sim.base_start(),
                      torch.ones(sim.worlds, 1, device=q.device)), -1)


def placement(sim: Sim, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Held-cube error to the base top and base displacement, both ``(worlds,)`` metres."""
    held, base = q[sim.held, :3], q[sim.base, :3]
    stack = base + torch.tensor([0.0, 0.0, sim.gap], device=q.device)
    return (held - stack).norm(dim=-1), (base - sim.base_start()).norm(dim=-1)


SUCCESS_ERROR = 0.02
SUCCESS_BASE_SHIFT = 0.02


def stacked(sim: Sim, q: torch.Tensor, qd: torch.Tensor) -> torch.Tensor:
    """Close to the base top, base not shoved, held cube at rest; ``(worlds,)`` bool."""
    error, shift = placement(sim, q)
    still = qd[sim.held, :3].norm(dim=-1) < 0.05
    return (error < SUCCESS_ERROR) & (shift < SUCCESS_BASE_SHIFT) & still


@torch.no_grad()
def evaluate(sim: Sim, act, held_offset: torch.Tensor, horizon: int) -> dict:
    """Run ``act(obs) -> (worlds, 4)`` from fixed starts and report placement and success."""
    q, qd = sim.state0(held_offset)
    for k in range(horizon):
        q, qd = step(sim, q, qd, act(observe(sim, q, qd)), k)
    error, shift = placement(sim, q)
    success = stacked(sim, q, qd)
    return {"mean_error_m": float(error.mean()), "median_error_m": float(error.median()),
            "mean_base_shift_m": float(shift.mean()), "success_rate": float(success.float().mean()),
            "starts": int(sim.worlds)}


def start_offsets(count: int, spread: float, generator: torch.Generator) -> torch.Tensor:
    """Held-cube offset from its scene pose: +-spread in x/y, lifted 6-16 cm."""
    u = torch.rand(count, 3, generator=generator)
    return torch.stack((spread * (2 * u[:, 0] - 1), spread * (2 * u[:, 1] - 1), 0.06 + 0.10 * u[:, 2]), -1)
