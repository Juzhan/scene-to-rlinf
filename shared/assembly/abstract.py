"""Abstract scene entities: spatial content without a model or physics.

A point, direction, rectangle or box is an ordinary top-level entry in the
scene ``entities`` list with ``kind: abstract``. It has a pose, can follow a
physical entity's root through ``mount`` and can be randomized by ``reset`` like
an asset. The scene only stores geometry and the initial placement; Task and
model code decide what an abstract entity means (target, workspace, keep-out
area, approach direction), and the Task may move it every environment step by
returning ``set`` (see ``python_task.Runtime``).

Poses use metres and wxyz quaternions in the local environment frame, or in the
mounted entity's root frame when mounted. Rectangle sizes are full X/Y lengths,
box sizes full X/Y/Z lengths. A direction is the local +X axis of its pose.
"""
from math import isfinite
from .information import Information

SHAPES = {'point': 0, 'direction': 0, 'rectangle': 2, 'box': 3}
AREAS = ('rectangle', 'box')
LABELS = {'point': '点', 'direction': '方向', 'rectangle': '矩形', 'box': '长方体'}


def check_geometry(name, shape, size):
    """Normalize one shape/size pair; sizes are positive full lengths."""
    if shape not in SHAPES:
        raise ValueError(f'Abstract entity {name}: shape must be point, direction, rectangle or box')
    if (not isinstance(size, (tuple, list)) or len(size) != SHAPES[shape] or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not isfinite(x) or x <= 0 for x in size)):
        raise ValueError(f'Abstract entity {name}: {shape} size requires {SHAPES[shape]} positive finite lengths')
    return tuple(float(x) for x in size)


def entity_key(settings, reference):
    """Scene path such as ``world/robot`` -> native key ``robot``."""
    root = settings.get('name', 'world') + '/'
    if not isinstance(reference, str) or not reference.startswith(root):
        raise ValueError(f'Entity reference must be a scene path below {root!r}: {reference!r}')
    return reference.removeprefix(root).replace('/', '__')


def abstract_entities(settings):
    """Top-level abstract entries by name, parsed without factories or a simulator."""
    from .layout import _entity
    result = {}
    for entry in settings.get('entities', ()):
        if isinstance(entry, dict) and entry.get('kind') == 'abstract':
            result[entry['name']] = _entity(entry, True, {})
    return result


def region_sources(settings):
    """Names of abstract entities used as some entity's ``reset.region``; regions stay static."""
    root = settings.get('name', 'world') + '/'
    found = set()

    def visit(value):
        if isinstance(value, dict):
            reset = value.get('reset')
            if isinstance(reset, dict) and isinstance(reset.get('region'), str):
                found.add(reset['region'].removeprefix(root))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(settings.get('entities', ()))
    return found


def commanded(settings):
    """Abstract entities held by a backend command term: everything not mounted and not a region.

    Their pose lives in an ``AbstractState``: reset samples it, the Task may move it.
    """
    regions = region_sources(settings)
    return {name: node for name, node in abstract_entities(settings).items()
            if node.mount is None and name not in regions}


def abstract_fields(node, held=False):
    """Public information of one abstract entity, all in the environment frame.

    ``held`` entities (see ``commanded``) also report the velocity of Task moves.
    """
    fields = {'position': Information('position', (3,), 'float', 'm', 'environment'),
              'orientation': Information('rotation', (4,), 'float', '1', 'environment', node.name, 'wxyz')}
    if node.shape == 'direction':
        fields['direction'] = Information('vector', (3,), 'float', '1', 'environment', axes=('xyz',))
    if node.size:
        fields['size'] = Information('number', (len(node.size),), 'float', 'm', node.name)
    if held:
        fields['linear_velocity'] = Information('vector', (3,), 'float', 'm/s', 'environment', axes=('xyz',))
    return fields


def abstract_catalog(settings):
    """``name.field`` -> (value, Information).

    A reset region's value is a constant list. Every other entity's value
    describes the live pose read by the backend each step: mounted entities
    follow their entity, ``commanded`` ones are held by the backend's command term.
    """
    result, held = {}, commanded(settings)
    for name, node in abstract_entities(settings).items():
        for field, info in abstract_fields(node, name in held).items():
            if field == 'size':
                value = list(node.size)
            elif name in held or node.mount is not None:
                value = {'abstract': name, 'field': field, 'commanded': name in held,
                         'mount': None if node.mount is None else entity_key(settings, node.mount.entity),
                         'pose': {'position': list(node.pose.position), 'orientation': list(node.pose.orientation)}}
            else:
                value = list(pose_field(node.pose, field))
            result[name + '.' + field] = (value, info)
    return result


def pose_field(pose, field):
    """A field of a constant Pose, matching ``tensor_field`` for live poses."""
    if field == 'position':
        return pose.position
    if field == 'orientation':
        return pose.orientation
    w, x, y, z = pose.orientation
    return (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))


def tensor_field(pose, field):
    """Batched (N, 7) xyz+wxyz pose -> one public field."""
    if field == 'position':
        return pose[:, :3]
    if field == 'orientation':
        return pose[:, 3:7]
    w, x, y, z = pose[:, 3:7].unbind(-1)
    return __import__('torch').stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)


def quaternion_multiply(a, b):
    import torch
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)), -1)


def rotate(quaternion, vector):
    import torch
    xyz = quaternion[..., 1:]
    cross = 2 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def mounted_pose(position, orientation, local):
    """Compose a parent's batched root pose with an authored local pose."""
    offset = position.new_tensor(local['position']).expand_as(position)
    rotation = orientation.new_tensor(local['orientation']).expand_as(orientation)
    return __import__('torch').cat((position + rotate(orientation, offset),
                                    quaternion_multiply(orientation, rotation)), -1)


def sample_positions(area, count, *, device, dtype=None, inset=0., generator=None):
    """Uniform positions inside an oriented rectangle/box, in environment coordinates.

    ``area`` is ``{'shape', 'size', 'pose'}``. The inset shrinks every sampled
    axis by a fixed distance; it is a geometric margin, not collision checking.
    A rectangle's Z is exactly its plane.
    """
    import torch
    size = check_geometry('area', area['shape'], area['size'])
    if area['shape'] not in AREAS:
        raise ValueError('Random placement needs a rectangle or box')
    if not isfinite(inset) or inset < 0 or 2 * inset >= min(size):
        raise ValueError('Inset must leave a nonempty sampling area')
    size = torch.tensor(size, device=device, dtype=dtype) - 2 * inset
    local = torch.zeros((count, 3), device=device, dtype=dtype)
    local[:, :len(size)] = (torch.rand((count, len(size)), device=device, dtype=dtype,
                                      generator=generator) - .5) * size
    pose = area.get('pose', {})
    q = local.new_tensor(pose.get('orientation', (1., 0., 0., 0.))).expand(count, 4)
    return rotate(q, local) + local.new_tensor(pose.get('position', (0., 0., 0.)))


def sample_poses(rule, base, areas, count, *, device, dtype=None, generator=None):
    """Batched (count, 7) poses for one reset rule.

    ``base`` is the entity's default pose. Region placement uses
    area rotation * sampled local-Z yaw * base rotation; ``pose_range`` offsets
    the base pose; ``states`` picks one authored pose uniformly.
    """
    import torch
    position = torch.tensor(base['position'], device=device, dtype=dtype).repeat(count, 1)
    orientation = torch.tensor(base['orientation'], device=device, dtype=dtype).repeat(count, 1)

    def yaw(angle):
        return torch.stack((torch.cos(angle / 2), 0 * angle, 0 * angle, torch.sin(angle / 2)), -1)

    def uniform(bounds):
        return bounds[0] + (bounds[1] - bounds[0]) * torch.rand(count, device=device, dtype=dtype, generator=generator)

    if rule.get('region') is not None:
        area = areas[rule['region']]
        position = sample_positions(area, count, device=device, dtype=dtype,
                                    inset=rule.get('inset', 0.), generator=generator)
        area_q = position.new_tensor(area.get('pose', {}).get('orientation', (1., 0., 0., 0.))).expand(count, 4)
        orientation = quaternion_multiply(area_q, quaternion_multiply(yaw(uniform(rule.get('yaw_range', (0., 0.)))),
                                                                      orientation))
    elif rule.get('pose_range'):
        ranges = rule['pose_range']
        position = position + torch.stack([uniform(ranges.get(axis, (0., 0.))) for axis in 'xyz'], -1)
        roll, pitch, heading = (uniform(ranges.get(axis, (0., 0.))) for axis in ('roll', 'pitch', 'yaw'))
        zero = 0 * roll
        rx = torch.stack((torch.cos(roll / 2), torch.sin(roll / 2), zero, zero), -1)
        ry = torch.stack((torch.cos(pitch / 2), zero, torch.sin(pitch / 2), zero), -1)
        orientation = quaternion_multiply(quaternion_multiply(quaternion_multiply(yaw(heading), ry), rx), orientation)
    else:
        states = rule['states']
        table = torch.tensor([[*s['position'], *s.get('orientation', (1., 0., 0., 0.))] for s in states],
                             device=device, dtype=dtype)
        choice = torch.randint(len(states), (count,), device=device, generator=generator)
        return table[choice]
    return torch.cat((position, orientation), -1)


def area(node):
    """Serializable geometry of one abstract rectangle/box, for reset sampling."""
    return {'shape': node.shape, 'size': list(node.size),
            'pose': {'position': list(node.pose.position), 'orientation': list(node.pose.orientation)}}


def reset_areas(settings):
    """Every abstract rectangle/box keyed by its scene path (``world/<name>``)."""
    root = settings.get('name', 'world')
    return {f'{root}/{name}': area(node) for name, node in abstract_entities(settings).items() if node.shape in AREAS}


def commanded_entity(settings, node):
    """Serializable description of one abstract entity held by a backend command term."""
    from dataclasses import asdict
    return {'name': node.name, 'shape': node.shape, 'size': list(node.size), 'pose': asdict(node.pose),
            'reset': None if node.reset is None else asdict(node.reset),
            'mount': None if node.mount is None else entity_key(settings, node.mount.entity)}


class AbstractState:
    """Batched pose and velocity of one commanded abstract entity.

    Both backends' command terms hold one: ``resample`` runs on episode reset and every
    ``reset.interval``, ``advance`` in the native command update after Task evaluation and
    resets, before observations. ``advance`` applies the pose the Task set this step to rows
    not resampled since the last update; velocity is always displacement / step_dt, so it is
    zero when the Task does not move the entity and right after any reset.
    """

    def __init__(self, entity, areas, num_envs, device):
        import torch
        self.entity, self.areas = entity, areas
        base = entity['pose']
        self.pose = torch.tensor([*base['position'], *base['orientation']], device=device).repeat(num_envs, 1)
        self.velocity = torch.zeros((num_envs, 3), device=device)
        self.fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

    def resample(self, env_ids):
        import torch
        rule, base = self.entity['reset'], self.entity['pose']
        count = len(self.pose[env_ids])
        if rule is not None:
            self.pose[env_ids] = sample_poses(rule, base, self.areas, count, device=self.pose.device)
        else:
            self.pose[env_ids] = torch.tensor([*base['position'], *base['orientation']], device=self.pose.device)
        self.velocity[env_ids] = 0.
        self.fresh[env_ids] = True

    def advance(self, dt, value=None):
        """Apply a Task-set (N, 3) position or (N, 7) xyz + wxyz pose, validated by the Task runtime."""
        import torch
        pose = self.pose.clone()
        if value is not None:
            if value.shape[1] == 3:
                pose[:, :3] = value
            else:
                pose = torch.cat((value[:, :3], value[:, 3:7] / value[:, 3:7].norm(dim=-1, keepdim=True)), -1)
        moving = ~self.fresh
        self.velocity[moving] = (pose[moving, :3].to(self.pose.dtype) - self.pose[moving, :3]) / dt
        self.pose[moving] = pose[moving].to(self.pose.dtype)
        self.fresh[:] = False

    def read(self, field):
        # Copies: the state is updated in place, readers may keep what they read.
        if field == 'linear_velocity':
            return self.velocity.clone()
        return tensor_field(self.pose, field).clone()


# Markers of entities listed in ``visualization.abstract``, the same in every backend. They are
# drawn only in the preview window/video, with no physics, and are never model input.
MARKER_COLOR = (0.2, 0.6, 1.0)
POINT_RADIUS = 0.04
EDGE_WIDTH = 0.008
# A direction is an arrow along its local +X axis, starting at the entity position. Its shape
# follows MuJoCo's arrow: shaft radius ARROW_WIDTH, the last third a cone of radius 1.75 * ARROW_WIDTH.
ARROW_COLOR = (0.0, 1.0, 0.0)
ARROW_LENGTH = 0.3
ARROW_WIDTH = 0.015
ARROW_HEAD_LENGTH = 1 / 3
ARROW_HEAD_RADIUS = 1.75


def marker_edges(entity):
    """Wireframe edges of a rectangle/box as (centre offsets in the entity frame, axis indices), else None.

    Rectangles and boxes are wireframes in every backend: Isaac's RTX real-time
    rendering ignores surface opacity, so a solid volume would hide the robot inside it.
    Each edge is a cylinder of diameter ``EDGE_WIDTH`` spanning its side length plus ``EDGE_WIDTH`` so the corners close.
    """
    if entity['shape'] not in AREAS:
        return None
    size = list(entity['size']) + [0.] * (3 - len(entity['size']))
    offsets, indices = [], []
    for axis in range(3):
        if not size[axis]:
            continue
        others = [other for other in range(3) if other != axis]
        corners = [(a, b) for a in ((-.5, .5) if size[others[0]] else (0.,))
                   for b in ((-.5, .5) if size[others[1]] else (0.,))]
        for a, b in corners:
            offset = [0.] * 3
            offset[others[0]], offset[others[1]] = a * size[others[0]], b * size[others[1]]
            offsets.append(offset)
            indices.append(axis)
    return offsets, indices
