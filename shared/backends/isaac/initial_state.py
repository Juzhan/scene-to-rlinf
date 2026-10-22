"""Translate authored physical initial values into native Isaac asset defaults."""
from math import isfinite

FIELDS = {'linear_velocity': 'lin_vel', 'angular_velocity': 'ang_vel',
          'joint_positions': 'joint_pos', 'joint_velocities': 'joint_vel'}


def apply_initial_state(native, values, path):
    if not values:
        return
    if values.keys() - FIELDS.keys():
        raise NotImplementedError(f'Entity {path}: semantic initial_state is not a native simulator property; initialize custom state through a project task event')
    initial = getattr(native, 'init_state', None)
    for key, value in values.items():
        field = FIELDS[key]
        if initial is None or not hasattr(initial, field):
            raise ValueError(f'{path}: this asset does not support {key}')
        if key.startswith('joint_'):
            if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
                raise ValueError(f'{path}: {key} must map joint names/patterns to values')
            numbers = list(value.values())
        else:
            if not isinstance(value, (list, tuple)) or len(value) != 3:
                raise ValueError(f'{path}: {key} requires three values')
            numbers = value
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in numbers):
            raise ValueError(f'{path}: {key} requires finite numeric values')
        if key.startswith('joint_'):
            # The editor supplies the complete measured joint list. Native
            # regex defaults cannot be merged with exact names (overlap errors).
            if value:
                setattr(initial, field, {name: float(number) for name, number in value.items()})
        else:
            setattr(initial, field, tuple(float(number) for number in value))
