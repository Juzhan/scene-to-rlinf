"""Default preview camera: frame environment 0's authored scene and follow its first robot.

``view``/``record`` use this only when neither ``--camera-eye`` nor ``--camera-target`` is
given. The extent is read from the scene description (no simulator): robots, objects and
abstract regions at their authored poses; terrain, lights, sensors and global entities are
left out, so a large ground does not push the camera away. Each backend applies the result
with its own viewer (Isaac ``ViewerCfg``, MuJoCo camera), following the robot's root.
"""
import math

from .spec import Abstract

DIRECTION = (1., 1., .7)  # From the scene centre towards the eye, as Isaac's default view.
MINIMUM_DISTANCE = 2.5
MARGIN = .5  # Metres around entity origins: a pose is a root, not the body's extent.


def _points(description):
    from .catalog import _ASSETS, entity_category
    robots, points = [], []
    for path, (node, pose) in description.index().items():
        position = tuple(pose.position)
        if isinstance(node, Abstract):
            if node.mount is not None:
                continue  # Follows a body; its authored pose is relative to that root.
            if node.shape in ('rectangle', 'box'):
                half = [value / 2 for value in node.size[:2]]
                points += [(position[0] + x, position[1] + y, position[2])
                           for x in (-half[0], half[0]) for y in (-half[1], half[1])]
            else:
                points.append(position)
            continue
        if not node.factory or node.scope == 'global' or node.mount is not None:
            continue
        capabilities = _ASSETS.get(node.factory, (None, None, []))[2]
        category = entity_category(capabilities, node.kind, node.factory)
        if category in ('robot', 'object'):
            points.append(position)
            if category == 'robot':  # Native scene key, the same rule for every backend.
                robots.append((path.removeprefix(description.root.name + '/').replace('/', '__'), position))
    return robots, points


def preview_camera(description):
    """``eye``/``lookat`` (metres) and ``follow`` (robot scene key or None); empty without content.

    With ``follow`` the coordinates are relative to that robot's authored root position,
    otherwise to environment 0's origin, as the explicit preview options are."""
    robots, points = _points(description)
    if not points:
        return {}
    centre = [(min(axis) + max(axis)) / 2 for axis in zip(*points)]
    radius = max(math.dist(point, centre) for point in points) + MARGIN
    distance = max(MINIMUM_DISTANCE, 2.2 * radius)
    norm = math.hypot(*DIRECTION)
    eye = [c + distance * d / norm for c, d in zip(centre, DIRECTION)]
    follow, origin = (robots[0] if robots else (None, (0., 0., 0.)))
    relative = lambda point: tuple(round(p - o, 4) for p, o in zip(point, origin))
    return {'eye': relative(eye), 'lookat': relative(centre), 'follow': follow}


def complete_viewer(viewer, description):
    """The preview viewer with the default framing unless the user placed the camera."""
    if viewer is None or {'eye', 'lookat'} & viewer.keys():
        return viewer
    return {**viewer, **preview_camera(description)}
