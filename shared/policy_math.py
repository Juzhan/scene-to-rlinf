"""Compatibility forward: the helpers are part of shared.policy (model feature construction).

Kept for projects and frozen snapshots that import ``shared.policy_math``; new policy.py
files import inverse_rotate/yaw/relative_goal from shared.policy.
"""
from shared.policy import inverse_rotate, relative_goal, yaw  # noqa: F401 - re-exported
