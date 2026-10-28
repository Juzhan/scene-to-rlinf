"""Composition is a tree; physical connections are separate edges.

Distances are metres and quaternions use wxyz. Entity poses are relative to the
logical parent, for initial layout only. A Mount follows a live body/frame.
Factories are asset identifiers resolved by the backend's project registry.
"""

from dataclasses import dataclass, field, replace
from math import isfinite, sqrt
import re
import posixpath
from typing import ClassVar


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float] = (0., 0., 0.)
    orientation: tuple[float, float, float, float] = (1., 0., 0., 0.)

    def __post_init__(self):
        if len(self.position) != 3 or len(self.orientation) != 4:
            raise ValueError("Pose requires xyz position and wxyz quaternion")
        if not all(isfinite(x) for x in (*self.position, *self.orientation)):
            raise ValueError("Pose must be finite")
        norm = sqrt(sum(x*x for x in self.orientation))
        if abs(norm - 1) > 1e-5:
            raise ValueError("Pose quaternion must be normalized")
        object.__setattr__(self, "position", tuple(float(x) for x in self.position))
        object.__setattr__(self, "orientation", tuple(float(x) for x in self.orientation))

    def compose(self, child):
        def mul(a, b):
            w, x, y, z = a
            v, i, j, k = b
            return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                    w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)
        q = self.orientation
        rotated = mul(mul(q, (0., *child.position)), (q[0], -q[1], -q[2], -q[3]))[1:]
        return Pose(tuple(a+b for a, b in zip(self.position, rotated)), mul(q, child.orientation))


@dataclass(frozen=True)
class Mount:
    entity: str
    body: str = ""  # Relative body path; sensors may use native body regexes.

    def __post_init__(self):
        if (not isinstance(self.entity, str) or not self.entity
                or not isinstance(self.body, str) or self.body.startswith("/")
                or any(part in (".", "..", "") for part in self.body.split("/")) and self.body):
            raise ValueError("Mount requires an entity and a relative body path")
        try:
            re.compile(self.body)
        except re.error as error:
            raise ValueError(f"Invalid body pattern: {self.body}") from error


@dataclass
class Entity:
    name: str
    pose: Pose = field(default_factory=Pose)
    factory: str | None = None
    parameters: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    initial_state: dict = field(default_factory=dict)  # Semantic metadata; runtime support is backend-specific.
    children: tuple["Entity", ...] = ()
    mount: Mount | None = None  # For sensors only in the first Isaac backend.
    connections: tuple["FixedJoint", ...] = ()
    frames: tuple["BodyFrame", ...] = ()
    reset: "RootReset | None" = None
    scope: str | None = None  # environment or global; otherwise inherit the assembly/preset.
    prim_name: str | None = None  # Optional native prim leaf name; logical entity name stays stable.
    meshes: tuple[str, ...] = ()  # Scene references for native ray sensors.
    pose_authored: bool = True  # Loader distinguishes omitted poses from an explicit identity pose.
    kind: ClassVar[str] = "entity"


class Scene(Entity):
    """A reusable grouping of entities, with no active/passive assumption."""
    kind = "scene"


class Robot(Entity):
    """A role label; its native config owns articulations and actuators."""
    kind = "robot"


class Device(Entity):
    """A role for equipment; behavior is provided by project-owned systems."""
    kind = "device"


class Sensor(Entity):
    """A measurement-producing entity, independent of robot ownership."""
    kind = "sensor"


@dataclass
class Abstract(Entity):
    """Spatial scene content without a model or physics: point, direction, rectangle or box.

    Its pose is environment-local, or relative to a mounted entity's root. Task and
    model code decide what it means; the Task may move it (``set``), see ``abstract.py``.
    """
    shape: str = "point"
    size: tuple[float, ...] = ()
    kind: ClassVar[str] = "abstract"

    def __post_init__(self):
        from .abstract import check_geometry
        self.size = check_geometry(self.name, self.shape, self.size)


@dataclass(frozen=True)
class RootReset:
    states: tuple[Pose, ...] = ()
    pose_range: dict = field(default_factory=dict)
    region: str | None = None
    yaw_range: tuple[float, float] = (0., 0.)
    inset: float = 0.
    interval: float | None = None  # Abstract entities only: resample every interval seconds.

    def __post_init__(self):
        if sum((bool(self.states), bool(self.pose_range), self.region is not None)) != 1:
            raise ValueError("Reset requires exactly one of nonempty states, pose_range or region")
        if self.region is not None and (not isinstance(self.region, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z_0-9]*(/[A-Za-z_][A-Za-z_0-9]*)+", self.region)):
            raise ValueError("Reset region must be a scene path such as world/spawn_area")
        if self.interval is not None:
            if isinstance(self.interval, bool) or not isinstance(self.interval, (int, float)) or \
                    not isfinite(self.interval) or self.interval <= 0:
                raise ValueError("Reset interval must be a finite positive number of seconds")
            object.__setattr__(self, "interval", float(self.interval))
        if (not isinstance(self.yaw_range, (tuple, list)) or len(self.yaw_range) != 2 or
                not all(isinstance(x, (int, float)) and not isinstance(x, bool) and isfinite(x)
                        for x in self.yaw_range) or self.yaw_range[0] > self.yaw_range[1]):
            raise ValueError("Reset yaw_range requires finite ordered [min, max] radians")
        if not isinstance(self.inset, (int, float)) or isinstance(self.inset, bool) or not isfinite(self.inset) or self.inset < 0:
            raise ValueError("Reset inset must be finite and nonnegative")
        if self.region is None and (tuple(self.yaw_range) != (0., 0.) or self.inset != 0):
            raise ValueError("Reset yaw_range and inset require a region")
        object.__setattr__(self, "yaw_range", tuple(float(x) for x in self.yaw_range))
        if any(not isinstance(state, Pose) for state in self.states):
            raise ValueError("Reset states must contain Pose entries")
        object.__setattr__(self, "states", tuple(self.states))
        if not isinstance(self.pose_range, dict) or self.pose_range.keys() - {"x", "y", "z", "roll", "pitch", "yaw"}:
            raise ValueError("Invalid reset pose_range axes")
        normalized = {}
        for axis, bounds in self.pose_range.items():
            if (not isinstance(bounds, (tuple, list)) or len(bounds) != 2
                    or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                               and isfinite(value) for value in bounds) or bounds[0] > bounds[1]):
                raise ValueError(f"Reset pose_range.{axis} requires finite ordered [min, max]")
            normalized[axis] = tuple(float(value) for value in bounds)
        object.__setattr__(self, "pose_range", normalized)


@dataclass(frozen=True)
class BodyFrame:
    name: str
    body: str = ""
    pose: Pose = field(default_factory=Pose)

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.name):
            raise ValueError(f"Invalid body frame name: {self.name!r}")
        if not isinstance(self.body, str) or not isinstance(self.pose, Pose):
            raise ValueError("BodyFrame requires a body path string and Pose")
        Mount(".", self.body)
        _exact_body(self.body)


def _exact_body(body):
    if body and any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", part) for part in body.split("/")):
        raise ValueError("Physical connections and body frames require exact body paths, without wildcards")


@dataclass(frozen=True)
class FixedJoint:
    name: str
    child: Mount
    parent: Mount | None = None  # None means the local environment's world frame.
    parent_pose: Pose = field(default_factory=Pose)
    child_pose: Pose = field(default_factory=Pose)


@dataclass
class WorldSpec:
    root: Entity
    connections: tuple[FixedJoint, ...] = ()

    def resolve_mount(self, owner_path, mount):
        """Relative references make sensor-bearing subassemblies relocatable."""
        if mount.entity.startswith("."):
            entity = posixpath.normpath(posixpath.join(owner_path, mount.entity))
        else:
            entity = mount.entity
        return Mount(entity, mount.body)

    def _resolve_connections(self, index):
        connections = list(self.connections)
        for path, (node, _) in index.items():
            for joint in node.connections:
                if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", joint.name):
                    raise ValueError(f"Invalid internal joint name: {joint.name!r}")
                if joint.parent is None:
                    raise ValueError("Internal connections need a parent; world anchors belong to WorldSpec")
                mounts = []
                for mount in (joint.child, joint.parent):
                    if not mount.entity.startswith("."):
                        raise ValueError("Internal connections require relative entity references")
                    resolved = self.resolve_mount(path, mount)
                    if resolved.entity != path and not resolved.entity.startswith(path + "/"):
                        raise ValueError(f"Internal connection escapes its assembly: {path}")
                    mounts.append(resolved)
                connections.append(replace(joint, name=path.replace("/", "__") + "__" + joint.name,
                                           child=mounts[0], parent=mounts[1]))
        return tuple(connections)

    def index(self):
        result = {}
        seen = set()

        def walk(node, parent_path, pose):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", node.name):
                raise ValueError(f"Invalid entity name: {node.name!r}")
            path = f"{parent_path}/{node.name}" if parent_path else node.name
            if id(node) in seen:
                raise ValueError(f"Entity instance reused or cyclic: {path}; create a fresh instance")
            if path in result:
                raise ValueError(f"Duplicate entity path: {path}")
            seen.add(id(node))
            world_pose = pose.compose(node.pose)
            result[path] = (node, world_pose)
            for child in node.children:
                walk(child, path, world_pose)
        walk(self.root, "", Pose())

        def check_mount(mount):
            if mount.entity not in result or result[mount.entity][0].factory is None:
                raise ValueError(f"Mount must reference a concrete asset: {mount.entity}")

        for path, (node, _) in result.items():
            if isinstance(node, Abstract):
                self._check_abstract(path, node, result, check_mount)
                continue
            if node.scope not in (None, "environment", "global"):
                raise ValueError(f"Entity {path}: scope must be environment or global")
            if node.prim_name is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", node.prim_name):
                raise ValueError(f"Invalid prim_name at {path}: {node.prim_name!r}")
            if node.meshes and not isinstance(node, Sensor):
                raise ValueError(f"Mesh references require a sensor: {path}")
            for reference in node.meshes:
                check_mount(self.resolve_mount(path, Mount(reference)))
            if node.reset is not None and (not isinstance(node.reset, RootReset)
                                          or not node.factory or isinstance(node, Sensor)):
                raise ValueError(f"Reset requires a concrete non-sensor asset and RootReset: {path}")
            if node.reset is not None and node.reset.interval is not None:
                raise ValueError(f"Entity {path}: reset interval is only available on abstract entities; "
                                 "physical assets are placed at episode reset")
            self._check_reset_region(path, node, result)
            if node.frames and (not node.factory or isinstance(node, Sensor)):
                raise ValueError(f"Body frames require a concrete non-sensor asset: {path}")
            frame_names = set()
            for frame in node.frames:
                if not isinstance(frame, BodyFrame):
                    raise ValueError(f"Expected BodyFrame at {path}")
                if frame.name in frame_names:
                    raise ValueError(f"Duplicate body frame at {path}: {frame.name}")
                frame_names.add(frame.name)
            if node.mount:
                mount = self.resolve_mount(path, node.mount)
                check_mount(mount)
                if not isinstance(node, Sensor) or node.children:
                    raise ValueError("Live mounts require leaf sensors or abstract entities; "
                                     "use FixedJoint for physical attachments")
                if mount.entity == path or isinstance(result[mount.entity][0], Sensor):
                    raise ValueError("Sensor mount must refer to a non-sensor asset")
            if isinstance(node, Sensor) and not node.factory:
                raise ValueError(f"Sensor needs a config factory: {path}")
            if isinstance(node, Sensor) and node.children:
                raise ValueError(f"Sensor must be a leaf: {path}")
            for key in (*node.parameters, *node.initial_state):
                if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
                    raise ValueError(f"Invalid property name at {path}: {key!r}")

        joint_names, incoming = set(), {}
        for joint in self._resolve_connections(result):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", joint.name) or joint.name in joint_names:
                raise ValueError(f"Invalid or duplicate joint name: {joint.name}")
            joint_names.add(joint.name)
            check_mount(joint.child)
            _exact_body(joint.child.body)
            if isinstance(result[joint.child.entity][0], Sensor):
                raise ValueError("Sensor frames use mounts; sensor housings need a separate physical entity")
            if joint.parent:
                check_mount(joint.parent)
                _exact_body(joint.parent.body)
                if isinstance(result[joint.parent.entity][0], Sensor):
                    raise ValueError("Physical joint parent must not be a sensor")
            # Backends randomize roots first, then place attached children at their joint frames,
            # so only the root of an attachment chain may own a pose reset.
            if result[joint.child.entity][0].reset is not None:
                raise ValueError(f"{joint.child.entity}: an attached child is placed by its parent; "
                                 "remove its reset (randomize the root of the attachment chain instead)")
            child = joint.child.entity
            if child in incoming:
                raise ValueError(f"Multiple attachment parents for {child}")
            incoming[child] = joint.parent.entity if joint.parent else None
        for child in incoming:
            chain, current = set(), child
            while current in incoming:
                if current in chain:
                    raise ValueError("Cyclic physical attachment; closed loops need a backend-specific implementation")
                chain.add(current)
                current = incoming[current]
        return result

    def _check_reset_region(self, path, node, result):
        if node.reset is None or node.reset.region is None:
            return
        area = result.get(self.resolve_mount(path, Mount(node.reset.region)).entity, (None,))[0]
        if not isinstance(area, Abstract) or area.shape not in ("rectangle", "box"):
            raise ValueError(f"Entity {path}: reset region must be an abstract rectangle or box: {node.reset.region}")
        if area.mount is not None or area.reset is not None:
            raise ValueError(f"Entity {path}: reset region {node.reset.region} must be static "
                             "(not mounted or randomized itself)")
        if 2 * node.reset.inset >= min(area.size):
            raise ValueError(f"Entity {path}: reset inset consumes the whole region")

    def _check_abstract(self, path, node, result, check_mount):
        if posixpath.dirname(path) != self.root.name:
            raise ValueError(f"Abstract entity {path} must be a top-level scene entity")
        if (node.factory or node.children or node.connections or node.frames or node.meshes or node.prim_name
                or node.scope or node.parameters or node.initial_state):
            raise ValueError(f"Abstract entity {path} has no model, physics, children or native properties; "
                             "it only declares shape, size, pose, mount and reset")
        if node.mount is not None:
            mount = self.resolve_mount(path, node.mount)
            check_mount(mount)
            if mount.body:
                raise ValueError(f"Abstract entity {path} follows an entity root; body mounts are not supported")
            if isinstance(result[mount.entity][0], Sensor):
                raise ValueError(f"Abstract entity {path} must follow a physical entity, not a sensor")
            if node.reset is not None:
                raise ValueError(f"Abstract entity {path}: a mounted abstract entity cannot be randomized by reset")
        if node.reset is not None:
            if not isinstance(node.reset, RootReset):
                raise ValueError(f"Expected RootReset at {path}")
            self._check_reset_region(path, node, result)

    def ordered_connections(self):
        index = self.index()
        pending, done, ordered = list(self._resolve_connections(index)), set(), []
        children = {j.child.entity for j in pending}
        while pending:
            for joint in pending[:]:
                if joint.parent is None or joint.parent.entity not in children or joint.parent.entity in done:
                    ordered.append(joint)
                    done.add(joint.child.entity)
                    pending.remove(joint)
        return ordered
