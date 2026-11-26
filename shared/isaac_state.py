"""Compatibility forward for existing projects: see shared.backends.isaac.legacy.state.

New projects use shared.assembly.project (Project.register) and the training framework's
own subprocess adapter; nothing in the shared layers imports this module.
"""
from shared.backends.isaac.legacy.state import *  # noqa: F401,F403 - re-exported
from shared.backends.isaac.legacy.state import IsaacStateEnv, ProcessExit, make_simulation, resolve  # noqa: F401
