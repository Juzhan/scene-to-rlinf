"""Historical experiment drivers, kept runnable but outside the product surface.

Each module reproduces a recorded experiment (RLinf reference runs, SFT data
conversion, LeRobot/LIBERO probes, lift/stack ablations) and may import an old
environment or the training framework directly. New projects use ``run.py``
commands and the shared layers instead; see docs/PACKAGING_PLAN.md.
"""
