"""Backend-neutral domain randomization declared in an environment's ``init_params``.

``env.train.init_params.randomization`` (evaluation follows it unless ``env.eval`` names its own)
maps scene entities to physical properties; each backend's ``randomization`` hook binds them to
its native event terms::

    randomization:
      cube:
        mass: {scale: [0.5, 2.0]}            # or add / abs; optional bodies: [link, ...]
        friction: [0.4, 1.2]                 # sliding friction of every collision shape
      robot:
        mass: {bodies: [base], add: [-1.0, 3.0]}
        com: {bodies: [base], x: [-0.03, 0.03], y: [-0.03, 0.03]}
        push: {interval_s: [8, 12], velocity: {x: [-0.5, 0.5], y: [-0.5, 0.5]}}

Mass, centre of mass and friction are sampled once per environment when it is created;
pushes add a sampled root velocity at random intervals. Mass changes scale the inertia by the
same ratio. Friction is the entity's coefficient as a PhysX material: against a friction-1
partner the pair gets :func:`unit_contact_friction` of it under the scene's
``simulation.physics_material.friction_combine_mode``. Nothing here imports a simulator.
"""
from dataclasses import dataclass, field
from math import isfinite

PROPERTIES = ('mass', 'friction', 'com', 'push')
OPERATIONS = ('scale', 'add', 'abs')
DISTRIBUTIONS = ('uniform', 'log_uniform')
VELOCITY_AXES = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')
COM_AXES = ('x', 'y', 'z')
# PhysX friction of a material against the default friction-1 material, per combine mode.
COMBINE = {'average': lambda value: (value + 1) / 2, 'min': lambda value: min(value, 1.),
           'multiply': lambda value: value, 'max': lambda value: max(value, 1.)}


@dataclass
class Randomization:
    """One property of one entity; ``key`` is the backend scene key of ``path``.

    Not frozen: Isaac Lab's configclass rewrites fields of the settings it copies."""
    path: str
    key: str
    property: str
    bodies: tuple = ()
    operation: str = 'abs'
    distribution: str = 'uniform'
    range: tuple = ()
    axes: dict = field(default_factory=dict)
    interval_s: tuple = ()


def _range(where, value, *, positive=False):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) for x in value)):
        raise ValueError(f'{where}: expected [low, high] finite numbers, got {value!r}')
    low, high = float(value[0]), float(value[1])
    if high < low:
        raise ValueError(f'{where}: high {high} is below low {low}')
    if positive and low <= 0:
        raise ValueError(f'{where}: values must be positive, got {value!r}')
    return low, high


def _bodies(where, value):
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(b, str) and b for b in value):
        raise ValueError(f'{where}.bodies: list body names explicitly')
    if any(set(b) & set('.*[]()+?\\|^$') for b in value):
        raise ValueError(f'{where}.bodies: name each body; patterns are not accepted')
    return tuple(value)


def _axes(where, value, names):
    if not isinstance(value, dict) or not value or value.keys() - set(names):
        raise ValueError(f'{where}: expected a mapping of {list(names)} to [low, high]')
    return {axis: _range(f'{where}.{axis}', value[axis]) for axis in names if axis in value}


def _entity(name, index, root):
    if not isinstance(name, str) or not name:
        raise ValueError(f'randomization: entity names must be strings, got {name!r}')
    path = name if name.startswith(root + '/') else f'{root}/{name.replace(".", "/")}'
    if path not in index:
        raise ValueError(f'randomization.{name}: unknown entity {path}')
    node = index[path][0]
    if not node.factory or getattr(node, 'kind', None) in ('abstract', 'sensor'):
        raise ValueError(f'randomization.{name}: only physical entities (robots and objects) can be randomized')
    return path, path.removeprefix(root + '/').replace('/', '__')


def resolve(spec, description):
    """Validate a declaration against the scene; returns a tuple of :class:`Randomization`."""
    if spec is None:
        return ()
    if not isinstance(spec, dict):
        raise ValueError('randomization must map entity names to properties')
    index, root = description.index(), description.root.name
    result = []
    for name, properties in spec.items():
        path, key = _entity(name, index, root)
        where = f'randomization.{name}'
        if not isinstance(properties, dict) or not properties or properties.keys() - set(PROPERTIES):
            raise ValueError(f'{where}: expected a mapping with some of {list(PROPERTIES)}')
        for prop, value in properties.items():
            place = f'{where}.{prop}'
            if prop == 'friction':
                result.append(Randomization(path, key, prop, range=_range(place, value)))
                if result[-1].range[0] < 0:
                    raise ValueError(f'{place}: friction must be nonnegative')
                continue
            if not isinstance(value, dict):
                raise ValueError(f'{place}: expected a mapping')
            if prop == 'mass':
                operations = [op for op in OPERATIONS if op in value]
                if len(operations) != 1 or value.keys() - {*OPERATIONS, 'bodies', 'distribution'}:
                    raise ValueError(f'{place}: give exactly one of {list(OPERATIONS)}, optional bodies and distribution')
                operation = operations[0]
                distribution = value.get('distribution', 'uniform')
                if distribution not in DISTRIBUTIONS:
                    raise ValueError(f'{place}.distribution must be one of {list(DISTRIBUTIONS)}')
                bounds = _range(f'{place}.{operation}', value[operation],
                                positive=operation != 'add' or distribution == 'log_uniform')
                result.append(Randomization(path, key, prop, bodies=_bodies(place, value.get('bodies')),
                                            operation=operation, distribution=distribution, range=bounds))
            elif prop == 'com':
                if 'bodies' not in value:
                    raise ValueError(f'{place}: name the bodies whose centre of mass moves')
                result.append(Randomization(path, key, prop, bodies=_bodies(place, value['bodies']),
                                            operation='add', axes=_axes(place, {k: v for k, v in value.items()
                                                                                if k != 'bodies'}, COM_AXES)))
            else:  # push
                if value.keys() != {'interval_s', 'velocity'}:
                    raise ValueError(f'{place}: expected interval_s and velocity')
                interval = _range(f'{place}.interval_s', value['interval_s'], positive=True)
                result.append(Randomization(path, key, prop, axes=_axes(f'{place}.velocity', value['velocity'],
                                                                        VELOCITY_AXES), interval_s=interval))
    return tuple(result)


def unit_contact_friction(value, settings):
    """Pair coefficient against a friction-1 partner. MuJoCo-family engines have no combine mode:
    they give the entity's geoms contact priority and this coefficient, as PhysX would combine it."""
    mode = settings.get('simulation', {}).get('physics_material', {}).get('friction_combine_mode', 'average')
    return COMBINE[mode](value)


def unsupported(backend, randomizations):
    """Hook body for backends without native randomization: refuse instead of ignoring."""
    if randomizations:
        raise NotImplementedError(f'{backend} does not implement domain randomization '
                                  f'({sorted({r.property for r in randomizations})}); remove init_params.randomization')
