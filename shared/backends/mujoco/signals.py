"""Discover named signals from an existing robosuite/LIBERO scene.

Uses the shared observation selection API. This adapter does not load assets or
assume a particular task; raw() must return the latest simulator observations.
"""
from shared.assembly.signals import Signal, Signals
from shared.assembly.information import Information


def scene_signals(native, raw):
    signals = Signals()
    for name, obj in {**native.objects_dict, **native.fixtures_dict}.items():
        body = obj.root_body
        for field, method, unit, frame in (
            ('position', 'get_body_xpos', 'm', 'world'),
            ('orientation', 'get_body_xquat', '1', 'world'),
            ('linear_velocity', 'get_body_xvelp', 'm/s', 'world'),
            ('angular_velocity', 'get_body_xvelr', 'rad/s', 'world'),
        ):
            kind = 'position' if field == 'position' else 'rotation' if field == 'orientation' else 'vector'
            info = Information(kind, (4,) if kind == 'rotation' else (3,), 'float', unit, frame,
                               target_frame=body if kind == 'rotation' else None,
                               encoding='wxyz' if kind == 'rotation' else None)
            signals.register(f'entities/{name}/{field}', Signal(
                lambda method=method, body=body: getattr(native.sim.data, method)(body).copy(),
                category='entity', unit=unit, frame=frame, description=f'{name}: {field}',
                info=info, entity=name, source='simulation'))
    for index, robot in enumerate(native.robots):
        prefix = f'robot{index}'
        for field, suffix, unit, frame in (
            ('tcp_position', 'eef_pos', 'm', 'world'),
            ('tcp_orientation', 'eef_quat', '1', 'world'),
            ('joint_position', 'joint_pos', 'rad', None),
            ('joint_velocity', 'joint_vel', 'rad/s', None),
            ('gripper_position', 'gripper_qpos', 'm', None),
        ):
            key = f'{prefix}_{suffix}'
            if key in raw():
                kind = 'position' if field == 'tcp_position' else 'rotation' if field == 'tcp_orientation' else 'number'
                info = Information(kind, tuple(raw()[key].shape), 'float', unit, frame,
                                   target_frame=prefix + '/tcp' if kind == 'rotation' else None,
                                   encoding='xyzw' if kind == 'rotation' else None,
                                   axes=('joint',) if kind == 'number' else ())
                signals.register(f'robots/{prefix}/{field}', Signal(
                    lambda key=key: raw()[key].copy(), category='robot', unit=unit,
                    frame=frame, description=f'{prefix}: {field}', info=info, entity=prefix, source='simulation'))
    for camera in native.camera_names:
        key = camera + '_image'
        if key in raw():
            signals.register(f'cameras/{camera}/rgb', Signal(
                lambda key=key: raw()[key].copy(), category='camera',
                info=Information('image', tuple(raw()[key].shape), 'uint8', encoding='RGB', axes=('height', 'width', 'channel')),
                entity=camera, source='sensor', description=f'{camera}: raw RGB (orientation handled by policy processor)'))
    return signals
