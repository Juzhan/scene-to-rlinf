"""Compatibility forward for existing projects: see shared.backends.isaac.legacy.policies.

The Gaussian actor-critic for the old state-task recipes; new projects define their model in
policy.py and the shared PPO adapter adds the noise and value network.
"""
from shared.backends.isaac.legacy.policies import *  # noqa: F401,F403 - re-exported
