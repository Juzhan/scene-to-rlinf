"""Compatibility forward for existing projects: see shared.backends.isaac.legacy.rewards.

Native reward functions for Isaac Lab 2 task presets; new projects write rewards in task.py.
"""
from shared.backends.isaac.legacy.rewards import *  # noqa: F401,F403 - re-exported
