"""One joint order per asset, whatever order the engine keeps.

PhysX lists articulation joints breadth-first (Go2: 4 hips, 4 thighs, 4 calves), MuJoCo
depth-first (FL hip, thigh, calf, FR ...). The asset's ASSET.json ``joint_order`` fixes
the order every backend presents: joint readings, ``joint_names()``, ``joint_position()``
and Jacobian columns. Each backend finds that record for its entity and reorders on the
device with :func:`canonical_joints`; assets without a record keep the engine's order.
Control targets are matched by joint name and need no reordering.
"""

# Public readings with one column per joint.
JOINT_FIELDS = frozenset({'joint_position', 'joint_velocity', 'joint_effort',
                          'default_joint_position', 'default_joint_velocity'})


def canonical_joints(native, canonical, where):
    """``(names, index)``: joint names in the asset order and the native column of each.

    ``index`` is None when the engine already uses that order or ``canonical`` is None
    (no recorded order). Different joint sets are an error, never a partial mapping."""
    native = tuple(native)
    if canonical is None or tuple(canonical) == native:
        return native, None
    if sorted(canonical) != sorted(native):
        raise ValueError(f'{where}: engine joints {native} differ from the asset joint_order {tuple(canonical)}')
    return tuple(canonical), [native.index(name) for name in canonical]
