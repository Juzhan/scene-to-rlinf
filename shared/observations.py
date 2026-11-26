"""Compatibility forward for existing projects: see shared.backends.isaac.legacy.observations.

Native observation functions for Isaac Lab 2 task presets; new projects select observations
in scene.yaml and read them through the typed views generated in observation.py.
"""
from shared.backends.isaac.legacy.observations import *  # noqa: F401,F403 - re-exported
