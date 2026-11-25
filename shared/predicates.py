"""Relations between scene entities for task.py, after the predicates of BEHAVIOR / LIBERO task files.

Each function takes ``obs`` nodes and returns one value per environment. A node is an entity
(``position``/``orientation`` and ``bounds``), an abstract rectangle or cuboid (``size``), an abstract
point, or a bare [B, 3] position. A relation comes in two forms: a boolean (``inside``, ``on_top`` ...)
for success and failure, and a gap in metres (``inside_gap`` ...) that is 0 once it holds, to shape a
reward towards it.

An entity is its ``bounds`` box, turned with it; a rectangle reaches up and down without end. A thing
is ``inside`` another when its centre is within the other's box: coarse for hollow or concave things
(a basket, a shelf with levels), where an abstract cuboid mounted on the entity says what counts.
Nothing here reads contacts: ``on_top`` is geometric and does not tell whether the support carries the
item. Holding a relation for some time is TaskMemory.hold.
"""
import torch
from shared.assembly.abstract import rotate
from shared.policy_math import inverse_rotate


def box(node):
    """``(centre [B, 3], orientation [B, 4] wxyz, half extents [B, 3])`` of a node, environment frame."""
    position = node if isinstance(node, torch.Tensor) else node.position
    if not hasattr(node, 'orientation'):
        identity = torch.zeros(*position.shape[:-1], 4, device=position.device, dtype=position.dtype)
        identity[..., 0] = 1
        return position, identity, torch.zeros_like(position)
    orientation = node.orientation
    if hasattr(node, 'bounds'):
        low, high = node.bounds.unbind(-2)
        return position + rotate(orientation, (low + high) / 2), orientation, (high - low) / 2
    if hasattr(node, 'size'):
        half = node.size / 2
        if half.shape[-1] == 2:
            half = torch.cat((half, torch.full_like(half[..., :1], torch.inf)), -1)
        return position, orientation, half
    return position, orientation, torch.zeros_like(position)


def centre(node):
    """Centre [B, 3] of a node's box, environment frame (m)."""
    return box(node)[0]


def _height(orientation, half):
    """Vertical half extent [B] of a turned box."""
    axes = torch.eye(3, device=half.device, dtype=half.dtype).expand(*half.shape[:-1], 3, 3)
    up = rotate(orientation.unsqueeze(-2).expand(*axes.shape[:-1], 4), axes)[..., 2]
    return (up.abs() * half.nan_to_num(posinf=0.)).sum(-1)


def bottom(node):
    """Height [B] of the lowest point of a node's box (m)."""
    centre, orientation, half = box(node)
    return centre[..., 2] - _height(orientation, half)


def top(node):
    """Height [B] of the highest point of a node's box (m)."""
    centre, orientation, half = box(node)
    return centre[..., 2] + _height(orientation, half)


def distance(a, b):
    """Distance [B] between the centres of two nodes (m)."""
    return torch.linalg.vector_norm(centre(a) - centre(b), dim=-1)


def near(a, b, within):
    """Whether the centres of two nodes are at most ``within`` metres apart."""
    return distance(a, b) <= within


def inside_gap(item, container):
    """How far [B] the centre of ``item`` is outside the box of ``container`` (m); 0 inside."""
    position, orientation, half = box(container)
    local = inverse_rotate(centre(item) - position, orientation)
    return torch.linalg.vector_norm((local.abs() - half).clamp(min=0), dim=-1)


def inside(item, container, margin=0.):
    """Whether the centre of ``item`` is in the box of ``container``, at least ``margin`` metres from its faces."""
    position, orientation, half = box(container)
    local = inverse_rotate(centre(item) - position, orientation)
    return (local.abs() <= half - margin).all(-1)


def _on_top(item, support, tolerance):
    position, orientation, half = box(support)
    local = inverse_rotate(centre(item) - position, orientation)
    across = torch.linalg.vector_norm((local[..., :2].abs() - half[..., :2]).clamp(min=0), dim=-1)
    level = position[..., 2] + _height(orientation, half)
    return across, ((bottom(item) - level).abs() - tolerance).clamp(min=0)


def on_top_gap(item, support, tolerance=.02):
    """How far [B] ``item`` is from resting on ``support`` (m): its centre over the support's footprint
    and its lowest point within ``tolerance`` of the support's top; 0 once it does."""
    across, height = _on_top(item, support, tolerance)
    return torch.hypot(across, height)


def on_top(item, support, tolerance=.02):
    """Whether ``item`` rests on ``support``: centre over its footprint, lowest point within ``tolerance``
    metres of its top. Geometric only; whether the support carries the item is not checked."""
    across, height = _on_top(item, support, tolerance)
    return (across == 0) & (height == 0)


def _joint(node, joint):
    if joint is None or isinstance(joint, int):
        return joint
    names = node.joint_names()
    if joint not in names:
        raise ValueError(f'unknown joint {joint!r}; joints: {list(names)}')
    return names.index(joint)


def open_fraction(node, joint=None):
    """How far a joint has moved from its closed end to the other limit, 0 to 1: [B] for one joint
    (name or index), [B, J] for all. The closed end is the limit nearer the default position.
    Reads ``joint_position``, ``default_joint_position`` and ``joint_limits``."""
    index = _joint(node, joint)
    pick = (lambda value: value) if index is None else (lambda value: value[:, index])
    position, default, limits = pick(node.joint_position), pick(node.default_joint_position), pick(node.joint_limits)
    low, high = limits.unbind(-1)
    lower = (default - low).abs() <= (high - default).abs()
    closed, opened = torch.where(lower, low, high), torch.where(lower, high, low)
    span = opened - closed
    return ((position - closed) / torch.where(span == 0, 1, span)).clamp(0, 1)


def is_open(node, joint=None, fraction=.5):
    """Whether a joint (all joints when None: any of them) has moved at least ``fraction`` of its range
    from the closed end."""
    opened = open_fraction(node, joint) >= fraction
    return opened if joint is not None else opened.any(-1)


def is_closed(node, joint=None, fraction=.05):
    """Whether a joint (all joints when None: every one) is within ``fraction`` of its range of the closed end."""
    closed = open_fraction(node, joint) <= fraction
    return closed if joint is not None else closed.all(-1)


def tilt(node, axis=(0., 0., 1.)):
    """Angle [B] (rad) between the node's local ``axis`` and the world +z."""
    orientation = node.orientation
    local = torch.tensor(axis, device=orientation.device, dtype=orientation.dtype).expand(*orientation.shape[:-1], 3)
    up = rotate(orientation, local / torch.linalg.vector_norm(local, dim=-1, keepdim=True))
    return torch.acos(up[..., 2].clamp(-1, 1))


def upright(node, within=.3, axis=(0., 0., 1.)):
    """Whether the node's local ``axis`` points within ``within`` rad of the world +z."""
    return tilt(node, axis) <= within
