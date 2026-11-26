"""Compatibility forward for existing projects: see shared.backends.isaac.legacy.terrains.

Isaac Lab 2 terrain presets; new scenes use ``run.py asset terrain`` pieces as usd_object.
"""
from shared.backends.isaac.legacy.terrains import *  # noqa: F401,F403 - re-exported
