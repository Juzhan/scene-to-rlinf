"""Robot capabilities on observation views: one interface, implemented by each backend.

``obs.<robot>`` nodes of the generated types inherit :class:`RobotView`. Besides
the selected tensors they answer kinematic queries about the same robot, batched
over environments, in the environment frame (metres, wxyz quaternions):

* ``frame_pose(frame)`` -> ``(position ['B', 3], orientation ['B', 4])``
* ``jacobian(frame)`` -> ``['B', 6, J]``: rows vx vy vz wx wy wz at the frame
  point, columns the robot joints in ``joint_position`` order
* ``differential_ik(frame, position, orientation=None)`` -> ``['B', J]``
  absolute joint position targets after one damped least-squares step

``frame`` is a body of the robot or one of its named frames: defaults of the asset
(:data:`ASSET_FRAMES`, e.g. ``tcp``) plus ``frames`` declared on the entity, which add
or override them. The chain is whatever moves that frame, so no joint list is
configured. Named frames are also selectable observations,
``<robot>.<frame>.position|orientation``, read through the same interface. Queries read the current simulator state. They exist
only in the environment process (fixed components, decode/initial, task.py);
the trained component receives tensors only and gets an explicit error.

A backend subclasses :class:`Kinematics` and implements the five primitives;
frame offsets and the IK step are shared and may be overridden.
"""
from __future__ import annotations

from typing import TYPE_CHECKING
from .abstract import quaternion_multiply, rotate
from .objects import ObjectView
if TYPE_CHECKING:
    import torch  # imported where used: the editor generates types without torch


class RobotView(ObjectView):
    """A robot node of an observation: selected tensors plus kinematic queries."""

    def _kinematics(self):
        name = self._prefix[:-1]
        if self._robots is None:
            raise RuntimeError(
                f'{name}: robot methods are not available here. The trained component runs in the policy '
                'worker and receives observation tensors only; call them in a fixed component, '
                'decode_action/initial_action or task.py, or select the needed information as observations.')
        return self._robots(name)

    def joint_names(self) -> tuple[str, ...]:
        """Joint names in ``joint_position`` order."""
        return self._kinematics().joint_names()

    def frame_pose(self, frame: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Current ``(position ['B', 3] m, orientation ['B', 4] wxyz)`` of a frame, environment frame."""
        return self._kinematics().frame_pose(frame)

    def jacobian(self, frame: str) -> torch.Tensor:
        """Geometric Jacobian ['B', 6, J] of a frame point (linear rows first), environment axes."""
        return self._kinematics().jacobian(frame)

    def differential_ik(self, frame: str, position: torch.Tensor, orientation: torch.Tensor | None = None, *,
                        damping: float = .05, max_step: float | None = None) -> torch.Tensor:
        """Joint targets ['B', J] moving ``frame`` toward the target pose by one damped least-squares step.

        ``position`` ['B', 3] m and optional ``orientation`` ['B', 4] wxyz are in the environment frame;
        without orientation only the position is tracked. ``dq = Jᵀ(JJᵀ + damping²I)⁻¹e`` is added
        to the current joint positions; ``max_step`` (rad or m) scales dq so no joint moves further.
        Joints that do not move the frame keep their current value. Joint limits are not applied.
        """
        return self._kinematics().differential_ik(frame, position, orientation, damping=damping, max_step=max_step)


class Kinematics:
    """Batched queries of one robot; backends implement the primitives below."""

    def __init__(self, name, frames):
        self.name, self.frames = name, frames

    # Backend primitives -------------------------------------------------
    def joint_names(self):
        raise NotImplementedError

    def joint_position(self):
        """['B', J] current joint positions, the same tensor order as the joint_position reading."""
        raise NotImplementedError

    def body_names(self):
        raise NotImplementedError

    def body_pose(self, body):
        """(position ['B', 3] environment frame, orientation ['B', 4] wxyz) of a body origin."""
        raise NotImplementedError

    def body_jacobian(self, body):
        """['B', 6, J] at the body origin, world-aligned axes, columns in joint_position order."""
        raise NotImplementedError

    # Shared -------------------------------------------------------------
    def _frame(self, frame):
        if not isinstance(frame, str):
            raise TypeError(f'{self.name}: frame must be a body or frame name')
        if frame in self.frames:
            body, offset = self.frames[frame]
            return body or self.body_names()[0], offset
        if frame in self.body_names():
            return frame, None
        raise ValueError(f'{self.name}: unknown frame {frame!r}; frames {sorted(self.frames)}, '
                         f'bodies {list(self.body_names())}')

    def frame_pose(self, frame):
        body, offset = self._frame(frame)
        position, orientation = self.body_pose(body)
        if offset is None:
            return position.clone(), orientation.clone()
        local, rotation = (position.new_tensor(value).expand_as(reference)
                           for value, reference in zip(offset, (position, orientation)))
        return position + rotate(orientation, local), quaternion_multiply(orientation, rotation)

    def jacobian(self, frame):
        body, offset = self._frame(frame)
        value = self.body_jacobian(body).clone()
        if value.shape[1:] != (6, len(self.joint_names())):
            raise ValueError(f'{self.name}: backend Jacobian has shape {tuple(value.shape)}')
        if offset is not None:
            position, orientation = self.body_pose(body)
            value = move_point(value, rotate(orientation, position.new_tensor(offset[0]).expand_as(position)))
        return value

    def differential_ik(self, frame, position, orientation=None, *, damping=.05, max_step=None):
        import torch
        current, current_orientation = self.frame_pose(frame)
        jacobian = self.jacobian(frame)
        error = torch.as_tensor(position, dtype=current.dtype, device=current.device).expand_as(current) - current
        if orientation is None:
            jacobian = jacobian[:, :3]
        else:
            target = torch.as_tensor(orientation, dtype=current.dtype, device=current.device).expand_as(current_orientation)
            error = torch.cat((error, orientation_error(target, current_orientation)), -1)
        identity = torch.eye(jacobian.shape[1], dtype=jacobian.dtype, device=jacobian.device)
        system = jacobian @ jacobian.transpose(1, 2) + damping ** 2 * identity
        step = (jacobian.transpose(1, 2) @ torch.linalg.solve(system, error[..., None]))[..., 0]
        if max_step is not None:
            largest = step.abs().amax(-1, keepdim=True)
            step = step * torch.clamp(max_step / largest.clamp_min(1e-12), max=1.)
        return self.joint_position() + step


def move_point(jacobian, arm):
    """Jacobian of the point ``arm`` ['B', 3] (world axes) away from the reference point: v + w x arm."""
    import torch
    linear = torch.linalg.cross(jacobian[:, 3:].transpose(1, 2), arm[:, None].expand(-1, jacobian.shape[2], -1), dim=-1)
    return torch.cat((jacobian[:, :3] + linear.transpose(1, 2), jacobian[:, 3:]), 1)


def orientation_error(target, current):
    """Rotation vector ['B', 3] (rad, world axes) taking ``current`` to ``target``."""
    import torch
    target = target / target.norm(dim=-1, keepdim=True)
    inverse = current * current.new_tensor((1., -1., -1., -1.))
    delta = quaternion_multiply(target, inverse)
    delta = torch.where(delta[..., :1] < 0, -delta, delta)
    sine = delta[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, delta[..., :1])
    return delta[..., 1:] * torch.where(sine > 1e-9, angle / sine.clamp_min(1e-9), 2.)


# Frames every instance of a robot asset carries, (body, (position, wxyz)) relative to the body.
# Defined once with the asset so observations, rewards and actions of every project mean the same point.
ASSET_FRAMES = {
    # Isaac Lab's Franka end effector: between the fingertips, along panda_hand z.
    'franka_panda': {'tcp': ('panda_hand', ((0., 0., .1034), (1., 0., 0., 0.)))},
    # Finger pads end 0.136 m along gripper_base z; the point is centred between them, 6 mm inside.
    'piper_x': {'tcp': ('gripper_base', ((0., 0., .13), (1., 0., 0., 0.)))},
}
FRAME_FIELDS = ('position', 'orientation')


def robot_frames(world):
    """Robot key -> {frame name: (body, (position, orientation))} of a composed scene, without a simulator.

    ``world`` is the scene description with assembly files expanded. Asset defaults come
    first; ``frames`` on the robot entity add frames or replace a default of the same name.
    """
    from .entity_information import SIGNALS
    index = world.index()
    methods = {name for name in dir(RobotView) if not name.startswith('_')}
    result = {}
    for path, (node, _) in index.items():
        if node.kind != 'robot':
            continue
        frames = dict(ASSET_FRAMES.get(node.factory, {}))
        for frame in node.frames:
            frames[frame.name] = (frame.body, (tuple(frame.pose.position), tuple(frame.pose.orientation)))
        children = {child.name for child in node.children}
        for name in frames:
            # A frame is read as obs.<robot>.<frame>; it must not shadow a method, field or child entity.
            if name in methods or name in SIGNALS or name in children or '__' in name:
                raise ValueError(f'{path}: frame name {name!r} clashes with a robot method, information field '
                                 'or child entity; rename the frame')
        result[path.split('/', 1)[1].replace('/', '__')] = frames
    return result


def frame_observations(frames):
    """Observation source -> (robot key, frame, field) for the named frames of ``robot_frames``."""
    from .entity_information import signal_id
    return {signal_id(robot + '__' + name, field): (robot, name, field)
            for robot, named in frames.items() for name in named for field in FRAME_FIELDS}


def provider(env, settings, kinematics_type):
    """Robot key -> cached backend Kinematics bound to one native environment.

    ``settings['_robot_frames']`` is filled by prepare_scene from the composed scene.
    """
    frames = settings.get('_robot_frames')
    cache = getattr(env, '_assembly_robots', None)
    if cache is None:
        cache = {}
        env._assembly_robots = cache

    def robot(name):
        if frames is None:
            raise ValueError('robot frames are resolved by prepare_scene; settings were not prepared')
        if name not in frames:
            raise ValueError(f'{name} is not a robot entity (kind: robot) of this scene')
        if name not in cache:
            cache[name] = kinematics_type(env, name, frames[name])
        return cache[name]
    return robot
