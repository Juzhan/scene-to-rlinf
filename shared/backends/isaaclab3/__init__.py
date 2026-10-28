"""Isaac Lab 3.0 scene backend, kitless on Newton (MuJoCo-Warp, default) or OvPhysX. SDK imports are deferred.

``env.*.init_params.physics`` selects the engine through Isaac Lab 3's own physics presets, and
``env.*.init_params.renderer`` the preview/camera renderer (Newton by default, or OVRTX).

Native Isaac Lab 3 managers own stepping and resets, as in the ``isaac`` backend; this package
binds the same shared scene/control/Task contract to its xyzw quaternions and warp-backed data.
"""

# run.py restarts under this workspace interpreter when the current one lacks the SDK.
SDK = "isaaclab_newton"
INTERPRETER = ".venv-il3/bin/python"
