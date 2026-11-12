"""MuJoCo-Warp contact capacity sized from the built Newton model, not from scene configuration.

The locomotion preset fixes 300 contacts and 1000 constraint rows per world. Scenes with many
convex pieces (SimReady cabinets) overflow that, while Newton's own estimate counts every
potentially touching shape pair and can exhaust GPU memory at thousands of worlds. The capacity
here takes Newton's per-world estimate, never below the preset, and bounds the totals across
worlds; ``simulation.newton.solver`` values still win when a scene sets them.
"""
import math

from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

PRESET_CONTACTS, PRESET_ROWS = 300, 1000
# Totals over all worlds; a pyramidal friction cone takes 2 * (condim - 1) = 4 rows per contact.
CONTACT_BUDGET, ROW_BUDGET, ROWS_PER_CONTACT = 1 << 19, 1 << 22, 4


def capacity(estimated_contacts, worlds, dofs_per_world):
    """Per-world (nconmax, njmax) from the model's global contact estimate."""
    worlds = max(1, worlds)
    contacts = max(PRESET_CONTACTS, min(math.ceil(estimated_contacts / worlds), CONTACT_BUDGET // worlds))
    # Joint limits may add two rows per degree of freedom on top of the contacts.
    rows = max(PRESET_ROWS, min(contacts * ROWS_PER_CONTACT + 2 * dofs_per_world, ROW_BUDGET // worlds))
    return contacts, rows


class NewtonSizedMJWarpManager(NewtonMJWarpManager):
    """Isaac Lab 3's MJWarp manager; ``None`` capacities are sized from the finalized model.

    Isaac Lab picks its Newton backend by the manager class name prefix ``newton``."""

    @classmethod
    def _create_solver(cls, model, solver_cfg):
        if solver_cfg.nconmax is None or solver_cfg.njmax is None:
            from newton._src.sim.collide import _estimate_rigid_contact_max
            solver_cfg = solver_cfg.copy()
            worlds = model.world_count
            contacts, rows = capacity(model.rigid_contact_max or _estimate_rigid_contact_max(model), worlds,
                                      math.ceil(model.joint_dof_count / max(1, worlds)))
            if solver_cfg.nconmax is None:
                solver_cfg.nconmax = contacts
            if solver_cfg.njmax is None:
                solver_cfg.njmax = rows
        return super()._create_solver(model, solver_cfg)
