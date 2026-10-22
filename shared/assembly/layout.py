"""Load entity scenes and reusable assemblies without starting a simulator."""

from copy import deepcopy
from pathlib import Path
import re

from .assets import factories_for
from .spec import Abstract, BodyFrame, Device, Entity, FixedJoint, Mount, Pose, Robot, RootReset, Scene, Sensor, WorldSpec


def _fields(value, allowed, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    unknown = value.keys() - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {label} fields: {sorted(unknown)}")
    return value


def _connection(data):
    data = _fields(data, ("name", "child", "parent", "parent_pose", "child_pose"), "connection")
    return FixedJoint(data["name"], child=Mount(**data["child"]),
                      parent=Mount(**data["parent"]) if data.get("parent") is not None else None,
                      parent_pose=Pose(**data.get("parent_pose", {})), child_pose=Pose(**data.get("child_pose", {})))


def _root_reset(data):
    data = _fields(data, ("states", "pose_range", "region", "yaw_range", "inset", "interval"), "reset")
    states = []
    for entry in data.get("states", ()):
        entry = _fields(entry, ("position", "orientation"), "reset state")
        if "position" not in entry:
            raise ValueError("Reset state requires an environment-local position")
        states.append(Pose(**entry))
    return RootReset(tuple(states), data.get("pose_range", {}), data.get("region"),
                     data.get("yaw_range", (0., 0.)), data.get("inset", 0.), data.get("interval"))


def _expand_entity(data, directory, stack=()):
    """Expand source-relative files before filtering cameras or patching children."""
    data = _fields(data, ("name", "kind", "factory", "assembly", "pose", "parameters", "tags",
                          "initial_state", "children", "mount", "connections", "frames", "reset", "camera_only",
                          "overrides", "connection_overrides", "scope", "meshes", "prim_name", "shape", "size"), "entity")
    if "assembly" in data:
        _fields(data, ("name", "assembly", "pose", "parameters", "camera_only", "overrides",
                       "connection_overrides", "scope", "prim_name"), "assembly instance")
        if not isinstance(data["assembly"], str) or not data["assembly"]:
            raise ValueError("assembly must be a nonempty file path")
        overrides = {key: value for key, value in data.items() if key != "assembly"}
        return _assembly_data(directory / data["assembly"], overrides, stack)
    if "overrides" in data or "connection_overrides" in data:
        raise ValueError("overrides and connection_overrides require an assembly instance")
    expanded = deepcopy(data)
    if data.get("kind") != "abstract":
        expanded["children"] = [_expand_entity(child, directory, stack) for child in data.get("children", ())]
    return expanded


def _instance_overrides(root, changes, *, connections=False):
    """Patch named components, keeping topology and relative references explicit."""
    from omegaconf import OmegaConf

    label = "connection_overrides" if connections else "overrides"
    if not isinstance(changes, dict):
        raise ValueError(f"{label} must map relative names to field mappings")
    if not changes:
        return
    targets = {}

    def index(node, path):
        if connections:
            for joint in node.get("connections", ()):
                key = f"{path}/{joint['name']}" if path else joint["name"]
                if key in targets:
                    raise ValueError(f"Duplicate connection override target: {key}")
                targets[key] = joint
        else:
            key = path or "."
            if key in targets:
                raise ValueError(f"Duplicate entity override target: {key}")
            targets[key] = node
        for child in node.get("children", ()):
            index(child, f"{path}/{child['name']}" if path else child["name"])

    index(root, "")
    allowed = ("parent_pose", "child_pose", "parent", "child") if connections else (
        "pose", "parameters", "tags", "initial_state", "mount", "scope", "meshes", "prim_name", "reset")
    for path, values in changes.items():
        if not isinstance(path, str) or not (path == "." and not connections or
                re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(/[A-Za-z_][A-Za-z_0-9]*)*", path)):
            raise ValueError(f"{label} target must be a relative component path: {path!r}")
        if path not in targets:
            raise ValueError(f"Unknown {label} target {path!r}; available: {', '.join(targets)}")
        values = _fields(values, allowed, f"{label} {path}")
        if connections and "parent" in values and values["parent"] is None:
            raise ValueError("Internal connection overrides require a parent")
        target = targets[path]
        # Merge only the changed fields; replacing a node's children would make
        # the other named targets stale and is deliberately not supported.
        merged = OmegaConf.to_container(OmegaConf.merge(
            {key: target[key] for key in values if key in target}, values), resolve=True)
        target.update(merged)
        if not connections and "reset" in values:
            # Reset modes are mutually exclusive; choosing a region must not
            # retain an inherited states/pose_range rule from the assembly.
            target["reset"] = deepcopy(values["reset"])


def _assembly_data(path, overrides, stack):
    from omegaconf import OmegaConf

    path = Path(path).resolve()
    if path in stack:
        raise ValueError(f"Cyclic assembly reference: {' -> '.join(str(item) for item in (*stack, path))}")
    data = _read_config(path, ("root",), "assembly")
    overrides = _fields(overrides if overrides is not None else {},
        ("name", "pose", "parameters", "camera_only", "overrides", "connection_overrides", "scope", "prim_name"), "assembly override")
    try:
        root = _expand_entity(data["root"], path.parent, (*stack, path))
        root_fields = {key: value for key, value in overrides.items()
                       if key not in ("overrides", "connection_overrides")}
        root = OmegaConf.to_container(OmegaConf.merge(root, root_fields), resolve=True)
        _instance_overrides(root, overrides.get("overrides", {}))
        _instance_overrides(root, overrides.get("connection_overrides", {}), connections=True)
        return root
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def _entity(data, cameras, factories):
    if not isinstance(data.get("camera_only", False), bool):
        raise ValueError("camera_only must be a boolean")
    if (data.get("camera_only", False) or data.get("factory") == 'camera') and not cameras:
        return None
    pose = Pose(**data.get("pose", {}))
    roles = {"entity": Entity, "scene": Scene, "device": Device, "robot": Robot, "sensor": Sensor}
    kind = data.get("kind", "entity")
    if kind == "abstract":
        _fields(data, ("name", "kind", "shape", "size", "pose", "mount", "reset", "tags"), f"abstract entity {data.get('name')}")
        if "shape" not in data:
            raise ValueError(f"Abstract entity {data.get('name')}: shape is required (point, direction, rectangle or box)")
        return Abstract(data["name"], pose=pose, tags=tuple(data.get("tags", ())),
                        mount=Mount(**data["mount"]) if data.get("mount") is not None else None,
                        reset=_root_reset(data["reset"]) if "reset" in data else None,
                        pose_authored="pose" in data, shape=data["shape"], size=tuple(data.get("size", ())))
    if "shape" in data or "size" in data:
        raise ValueError(f"Entity {data.get('name')}: shape and size belong to abstract entities (kind: abstract)")
    if kind not in roles:
        raise ValueError(f"Unknown entity kind: {kind}")
    factory = data.get("factory")
    if factory is not None and factory not in factories:
        raise ValueError(f"Unknown project asset factory: {factory}")
    children = tuple(node for child in data.get("children", ())
                     if (node := _entity(child, cameras, factories)) is not None)
    frames = []
    for entry in data.get("frames", ()):
        entry = _fields(entry, ("name", "body", "pose"), "body frame")
        frames.append(BodyFrame(entry["name"], entry.get("body", ""), Pose(**entry.get("pose", {}))))
    return roles[kind](data["name"], pose=pose, factory=factory,
                       parameters=data.get("parameters", {}), tags=tuple(data.get("tags", ())),
                       initial_state=data.get("initial_state", {}), children=children,
                       mount=Mount(**data["mount"]) if data.get("mount") is not None else None,
                       connections=tuple(_connection(joint) for joint in data.get("connections", ())),
                       frames=tuple(frames), reset=_root_reset(data["reset"]) if "reset" in data else None,
                       scope=data.get("scope"), prim_name=data.get("prim_name"), meshes=tuple(data.get("meshes", ())),
                       pose_authored="pose" in data)


def _read_config(path, fields, label):
    from omegaconf import OmegaConf

    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    data = _fields(data, ("schema_version", *fields), label)
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError(f"Unsupported {label} schema_version")
    return data


def load_assembly(path, cameras=False, *, overrides=None, factories=None, _stack=()):
    root = _assembly_data(path, overrides, _stack)
    return _entity(root, cameras, factories_for() if factories is None else factories)


def load_scene_config(path):
    data = _read_config(Path(path).resolve(), ("name", "entities", "connections", "observations",
                        "application", "simulation", "world", "actions", "task", "targets", "regions", "authoring", "control", "variables", "visualization"), "scene")
    if "regions" in data:
        raise ValueError(f"{path}: the regions section was replaced by abstract entities; "
                         "declare each region in entities with kind: abstract")
    return data


def load_world_spec(path, cameras=False, *, factories=None, backend='isaac'):
    path = Path(path).resolve()
    data = load_scene_config(path)
    return world_spec_from_config(data, directory=path.parent, cameras=cameras,
                                  factories=factories, backend=backend)


def world_spec_from_config(data, *, directory, cameras=False, factories=None, backend='isaac'):
    """Build an editable in-memory scene's description using its source folder."""
    factories = factories_for(backend) if factories is None else factories
    children = tuple(node for entry in data["entities"]
                     if (node := _entity(_expand_entity(entry, Path(directory)), cameras, factories)) is not None)
    if "regions" in data:
        raise ValueError("The regions section was replaced by abstract entities; "
                         "declare each region in entities with kind: abstract")
    description = WorldSpec(Scene(data["name"], children=children),
                            tuple(_connection(joint) for joint in data.get("connections", ())))
    description.index()
    return description


def load_observation_spec(path, profile="state"):
    from omegaconf import OmegaConf
    from .signals import Observation

    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    profiles = config.get("observations", {})
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"Unknown observation profile: {profile}")
    return Observation(profiles[profile])
