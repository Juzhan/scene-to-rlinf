"""Scene motion: how a kinematic or abstract entity moves by itself, declared in scene.yaml.

A pet roaming a room, a car on a road or a moving crate is part of the scene, not of the task::

    motion: {type: wander, region: world/crate_area, speed: [0.05, 0.15]}
    motion: {type: path, points: [[0, 0, 0], [2, 0, 0], [2, 1, 0]], speed: 0.4, loop: true, face: true}

``wander`` goes straight on inside an abstract rectangle/box at a speed drawn once per episode, in a
random direction, and turns back past its edges with the acceleration ``turn`` (m/s^2; default: the
top speed overshoots by 2 cm). ``region`` defaults to the entity's ``reset.region``. It moves in the
plane of the region, keeping the height above it.
``path`` follows straight segments through ``points`` (environment frame, m) at ``speed`` (m/s, or a
[min, max] range drawn per episode): ``loop: true`` closes the polyline and goes round, ``false`` goes
back and forth. Each reset puts the entity on the path, at its first point or (``start: random``) at a
random place along it, so a path replaces ``reset``.
``face: true`` turns the entity about the vertical axis to look along its motion: its authored
orientation is the one facing environment +X.
``custom`` runs a function of the project (an NPC that keeps away from the robot, a cart that stops for
it)::

    motion: {type: custom, function: .behaviors:keep_away, speed: 0.3}

``function`` names ``module:attribute``, relative to the project package when it starts with a dot. It
is called once per environment step for all environments as ``function(scene, motion)``: ``scene`` is
the Task's SceneInfo view (everything observable in the scene), ``motion`` a ``MotionContext`` with
the entity's current ``pose`` (B, 7), ``velocity`` (B, 3), ``dt``, ``active`` rows (not reset this
step), the other keys of the mapping as ``parameters`` and an episode-local ``memory`` (TaskMemory). It
returns the next (B, 3) position or (B, 7) pose, a batched tensor on the scene's device. A class is
instantiated once with ``parameters``; its ``step(scene, motion)`` is called and ``reset(env_ids)``,
when defined, at episode resets. Reset places the entity as for any other entity.

Poses are environment-local xyz + wxyz. The backend's command term holding the entity (the same one
that applies the Task's ``set``) calls ``reset`` on episode reset, ``start`` for the pose a reset
writes, and ``step`` once per environment step after resets and before observations; a pose the Task
sets that step wins over the motion. Velocity follows as displacement / step_dt, as for ``set``.
"""
from math import isfinite

TYPES = ('wander', 'path', 'custom')
OVERSHOOT = .02  # m past the region edge at the top speed, for the default wander turn
FIELDS = {'wander': {'type', 'region', 'speed', 'turn', 'face'},
          'path': {'type', 'points', 'speed', 'loop', 'start', 'face'}}


def _number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def check(name, motion, *, reset_region=None, has_reset=False):
    """Normalized copy of one entity's ``motion`` mapping; raises ValueError naming the entity."""
    def fail(message):
        raise ValueError(f'Entity {name}: motion {message}')
    if not isinstance(motion, dict) or motion.get('type') not in TYPES:
        fail(f'is a mapping with type {" or ".join(TYPES)}')
    kind = motion['type']
    if kind == 'custom':
        function = motion.get('function')
        if not isinstance(function, str) or function.count(':') != 1 or not all(function.split(':')):
            fail('custom names its function as module:attribute (e.g. .behaviors:keep_away)')
        return {'type': 'custom', 'function': function,
                'parameters': {key: value for key, value in motion.items() if key not in ('type', 'function')}}
    unknown = set(motion) - FIELDS[kind]
    if unknown:
        fail(f'{kind} takes {", ".join(sorted(FIELDS[kind] - {"type"}))}; unknown: {", ".join(sorted(unknown))}')
    speed = motion.get('speed')
    speed = [speed, speed] if _number(speed) else speed
    if (not isinstance(speed, (list, tuple)) or len(speed) != 2 or not all(map(_number, speed))
            or speed[0] < 0 or speed[1] <= 0 or speed[0] > speed[1]):
        fail('speed is a positive number or a [min, max] range (m/s)')
    face = motion.get('face', False)
    if not isinstance(face, bool):
        fail('face is true or false')
    result = {'type': kind, 'speed': [float(speed[0]), float(speed[1])], 'face': face}
    if kind == 'wander':
        region = motion.get('region', reset_region)
        if not isinstance(region, str):
            fail('wander needs a region (an abstract rectangle or box, e.g. world/area) or a reset.region')
        turn = motion.get('turn', result['speed'][1] ** 2 / (2 * OVERSHOOT))
        if not _number(turn) or turn <= 0:
            fail('turn is a positive acceleration (m/s^2)')
        result.update(region=region, turn=float(turn))
        return result
    if has_reset:
        fail('path places the entity on the path at every reset; remove its reset')
    points = motion.get('points')
    if (not isinstance(points, (list, tuple)) or len(points) < 2
            or any(not isinstance(p, (list, tuple)) or len(p) != 3 or not all(map(_number, p)) for p in points)):
        fail('path needs at least two points [x, y, z] (m, environment frame)')
    loop, start = motion.get('loop', True), motion.get('start', 'first')
    if not isinstance(loop, bool):
        fail('loop is true (go round) or false (back and forth)')
    if start not in ('first', 'random'):
        fail('start is first or random')
    points = [[float(v) for v in p] for p in points]
    closed = points + [points[0]] if loop else points
    if not sum(sum((a - b) ** 2 for a, b in zip(p, q)) ** .5 for p, q in zip(closed, closed[1:])) > 0:
        fail('path has zero length')
    result.update(points=points, loop=loop, start=start)
    return result


def motion_regions(settings):
    """Names of the abstract entities used as a wander region; regions stay static."""
    root = settings.get('name', 'world') + '/'
    found = set()
    for entry in settings.get('entities', ()) or ():
        stack = [entry]
        while stack:
            item = stack.pop()
            if not isinstance(item, dict):
                continue
            motion, reset = item.get('motion'), item.get('reset')
            if isinstance(motion, dict) and motion.get('type') == 'wander':
                region = motion.get('region', reset.get('region') if isinstance(reset, dict) else None)
                if isinstance(region, str):
                    found.add(region.removeprefix(root))
            stack.extend(item.get('children', ()) or ())
    return found


def entity_motions(settings):
    """Normalized motion of every entity declaring one, keyed by native key (``parent__child``)."""
    found = {}

    def visit(entries, prefix):
        for entry in entries or ():
            if not isinstance(entry, dict):
                continue
            key = prefix + entry['name']
            if entry.get('motion') is not None:
                reset = entry.get('reset') if isinstance(entry.get('reset'), dict) else None
                found[key] = check(key, entry['motion'], reset_region=(reset or {}).get('region'),
                                   has_reset=reset is not None)
            if entry.get('kind') != 'abstract':
                visit(entry.get('children'), key + '__')
    visit(settings.get('entities'), '')
    return found


def _yaw(angle):
    import torch
    return torch.stack((torch.cos(angle / 2), 0 * angle, 0 * angle, torch.sin(angle / 2)), -1)


class MotionContext:
    """What a custom motion function receives besides the scene (see the module docstring)."""

    def __init__(self, name, parameters, memory, num_envs, device):
        self.name, self.parameters, self.memory = name, parameters, memory
        self.num_envs, self.device = num_envs, device
        self.pose = self.velocity = self.active = None
        self.dt = None


def load_function(reference, package):
    """The object ``module:attribute`` names; a leading dot is relative to the project package."""
    from importlib import import_module
    module, attribute = reference.split(':')
    if module.startswith('.'):
        if not package:
            raise ValueError(f'motion function {reference}: a relative module needs the project package')
        module = package + module
    target = import_module(module)
    for part in attribute.split('.'):
        target = getattr(target, part)
    return target


class Motion:
    """Batched state of one entity's motion (one row per environment).

    ``base`` is the entity's authored pose (``{'position', 'orientation'}``); ``areas`` maps scene paths
    of abstract rectangles/boxes to ``{'shape', 'size', 'pose'}`` (``abstract.reset_areas``). A custom
    motion also needs the project ``package`` and ``scene``, a callable returning the SceneInfo view."""

    def __init__(self, motion, base, areas, num_envs, device, generator=None, *, package=None, scene=None, name=''):
        import torch
        self.motion, self.device, self.generator = motion, device, generator
        self.base = torch.tensor(base['orientation'], device=device, dtype=torch.float32)
        self.speed = torch.zeros(num_envs, device=device)
        if motion['type'] == 'custom':
            import inspect
            from .task_memory import TaskMemory
            target = load_function(motion['function'], package)
            instance = target(dict(motion['parameters'])) if inspect.isclass(target) else None
            self.function = instance.step if instance is not None else target
            self.custom_reset = getattr(instance, 'reset', None)
            self.context = MotionContext(name, dict(motion['parameters']), TaskMemory(), num_envs, device)
            self.scene, self.name = scene, name
            self.last = torch.zeros(num_envs, 3, device=device)
        elif motion['type'] == 'wander':
            area = areas.get(motion['region'])
            if area is None or area['shape'] not in ('rectangle', 'box'):
                raise ValueError(f'motion region {motion["region"]} must be an abstract rectangle or box')
            pose = area['pose']
            self.centre = torch.tensor(pose['position'], device=device, dtype=torch.float32)
            self.rotation = torch.tensor(pose['orientation'], device=device, dtype=torch.float32)
            self.half = torch.tensor(area['size'][:2], device=device, dtype=torch.float32) / 2
            self.velocity = torch.zeros(num_envs, 2, device=device)  # in the region's plane
            self.cruise = torch.zeros(num_envs, 2, device=device)  # per-axis speed drawn at reset
        else:
            points = torch.tensor(motion['points'], device=device, dtype=torch.float32)
            self.points = torch.cat((points, points[:1])) if motion['loop'] else points
            lengths = (self.points[1:] - self.points[:-1]).norm(dim=-1)
            self.ends = lengths.cumsum(0)
            self.lengths, self.total = lengths, float(self.ends[-1])
            self.distance = torch.zeros(num_envs, device=device)
            self.direction = torch.ones(num_envs, device=device)

    def _uniform(self, count, low, high):
        import torch
        return low + (high - low) * torch.rand(count, device=self.device, generator=self.generator)

    def reset(self, env_ids):
        """Draw this episode's speed (and wander direction or path start) for the given environments."""
        import math
        import torch
        if self.motion['type'] == 'custom':
            self.context.memory.reset(env_ids)
            self.last[env_ids] = 0.
            if self.custom_reset is not None:
                self.custom_reset(env_ids)
            return
        count = len(self.speed[env_ids])
        self.speed[env_ids] = self._uniform(count, *self.motion['speed'])
        if self.motion['type'] == 'wander':
            heading = self._uniform(count, 0., 2 * math.pi)
            self.velocity[env_ids] = self.speed[env_ids, None] * torch.stack((heading.cos(), heading.sin()), -1)
            self.cruise[env_ids] = self.velocity[env_ids].abs()
        else:
            self.direction[env_ids] = 1.
            self.distance[env_ids] = (self._uniform(count, 0., self.total) if self.motion['start'] == 'random'
                                      else torch.zeros(count, device=self.device))

    def start(self, env_ids):
        """(N, 7) pose a reset gives these environments, or None when ``reset`` places the entity (wander)."""
        if self.motion['type'] in ('wander', 'custom'):
            return None
        return self._on_path(self.distance[env_ids], self.direction[env_ids])

    def _on_path(self, distance, direction):
        import torch
        segment = torch.searchsorted(self.ends, distance.clamp(0., self.total), right=True).clamp(max=len(self.lengths) - 1)
        before = self.ends[segment] - self.lengths[segment]
        along = ((distance - before) / self.lengths[segment].clamp(min=1e-9)).clamp(0., 1.)[:, None]
        start, end = self.points[segment], self.points[segment + 1]
        position = start + along * (end - start)
        heading = (end - start)[:, :2] * direction[:, None]
        return torch.cat((position, self._orientation(heading, len(distance))), -1)

    def _orientation(self, heading, count, current=None):
        import torch
        from .abstract import quaternion_multiply
        if not self.motion['face']:
            return self.base.expand(count, 4) if current is None else current
        facing = quaternion_multiply(_yaw(torch.atan2(heading[:, 1], heading[:, 0])), self.base.expand(count, 4))
        if current is None:
            return facing
        return torch.where(heading.norm(dim=-1, keepdim=True) > 1e-9, facing, current)

    def step(self, pose, dt, active):
        """Next (B, 7) pose from the current (B, 7) ``pose``; only ``active`` rows advance their state."""
        import torch
        from .abstract import rotate
        pose = pose.to(torch.float32)
        count = len(pose)
        if self.motion['type'] == 'custom':
            return self._custom(pose, dt, active)
        if self.motion['type'] == 'wander':
            inverse = self.rotation * self.rotation.new_tensor((1., -1., -1., -1.))
            local = rotate(inverse.expand(count, 4), pose[:, :3] - self.centre)
            offset = local[:, :2]
            outside = offset.abs() > self.half
            back = self.motion['turn'] * dt * offset.sign() * outside
            # Inside, each axis runs at its drawn speed again: stepping the turn back would let it drift.
            cruise = torch.where(self.velocity >= 0, self.cruise, -self.cruise)
            velocity = torch.where(active[:, None], torch.where(outside, self.velocity - back, cruise), self.velocity)
            self.velocity = velocity
            local = torch.cat((offset + velocity * dt, local[:, 2:]), -1)
            position = rotate(self.rotation.expand(count, 4), local) + self.centre
            heading = rotate(self.rotation.expand(count, 4), torch.cat((velocity, velocity[:, :1] * 0), -1))[:, :2]
            return torch.cat((position, self._orientation(heading, count, pose[:, 3:7])), -1)
        distance = self.distance + self.direction * self.speed * dt
        direction = self.direction
        if self.motion['loop']:
            distance = distance.remainder(self.total)
        else:
            over, under = distance > self.total, distance < 0
            distance = torch.where(over, 2 * self.total - distance, torch.where(under, -distance, distance))
            direction = torch.where(over, -direction, torch.where(under, -direction, direction))
        self.distance = torch.where(active, distance, self.distance)
        self.direction = torch.where(active, direction, self.direction)
        return self._on_path(self.distance, self.direction)


    def _custom(self, pose, dt, active):
        import torch
        if self.scene is None:
            raise ValueError(f'{self.name}: a custom motion needs the scene view of the Task runtime')
        context = self.context
        context.pose, context.velocity, context.dt, context.active = pose.clone(), self.last.clone(), dt, active.clone()
        with torch.no_grad():
            result = self.function(self.scene(), context)
        where = f'{self.name}: motion function {self.motion["function"]}'
        if not torch.is_tensor(result) or not result.is_floating_point() or result.shape not in (
                (len(pose), 3), (len(pose), 7)):
            raise ValueError(f'{where} must return a float tensor of shape ({len(pose)}, 3) position or '
                             f'({len(pose)}, 7) xyz + wxyz, got {getattr(result, "shape", type(result).__name__)}')
        if result.device != pose.device:
            raise ValueError(f'{where} must return a tensor on {pose.device}, got {result.device}')
        if not torch.isfinite(result).all():
            raise ValueError(f'{where} returned non-finite values')
        result = result.to(torch.float32)
        if result.shape[1] == 3:
            result = torch.cat((result, pose[:, 3:7]), -1)
        else:
            norm = result[:, 3:7].norm(dim=-1, keepdim=True)
            if (norm < 1e-6).any():
                raise ValueError(f'{where} returned a zero quaternion')
            result = torch.cat((result[:, :3], result[:, 3:7] / norm), -1)
        result = torch.where(active[:, None], result, pose)
        self.last = torch.where(active[:, None], (result[:, :3] - pose[:, :3]) / dt, torch.zeros_like(self.last))
        return result
