"""Native scene keys, scopes and prim paths resolved from the shared entity tree.

Naming is shared by the USD-based backends: top-level names are native scene keys,
nested names use ``__``, and a mounted sensor's path denotes the observed body.
Nothing here imports a simulator SDK, so ``check`` and the editor resolve names
without launching one.
"""
from dataclasses import dataclass
import posixpath

from .assets import FACTORY_SCOPES
from .spec import Mount, Sensor


@dataclass(frozen=True)
class SceneMetadata:
    keys: dict[str, str]
    prim_paths: dict[str, str]
    scopes: dict[str, str]


def scene_metadata(description):
    """Resolve names and scopes without importing or launching a simulator.

    Top-level names are native scene keys. Nested names use ``__``. Logical
    parentage composes initial transforms, while a mount follows a live body.
    A mounted sensor's path denotes the observed body. Mounted cameras spawn
    independent prims and follow live physics through their native sensor update.
    """
    index = description.index()
    keys, paths, scopes = {}, {}, {}
    root = description.root.name
    for path, (node, _) in index.items():
        if node.factory:
            key = path.removeprefix(root + "/").replace("/", "__")
            if key in keys.values():
                raise ValueError(f"Native scene key collision at {path}: {key}")
            keys[path] = key

    def scope(path):
        if path in scopes:
            return scopes[path]
        node = index[path][0]
        parent = posixpath.dirname(path)
        if node.mount:
            mount = description.resolve_mount(path, node.mount)
            value = scope(mount.entity)
            if node.scope is not None and node.scope != value:
                raise ValueError(f"Entity {path}: mount scope must match {mount.entity} ({value})")
        elif node.scope is not None:
            value = node.scope
        else:
            value = FACTORY_SCOPES.get(node.factory, "environment")
            ancestor = parent
            while ancestor in index:
                if index[ancestor][0].scope is not None:
                    value = scope(ancestor)
                    break
                ancestor = posixpath.dirname(ancestor)
        scopes[path] = value
        return value

    for path, key in keys.items():
        node = index[path][0]
        prefix = "{ENV_REGEX_NS}" if scope(path) == "environment" else "/World"
        paths[path] = f"{prefix}/{node.prim_name or key}"
    for path in keys:
        node = index[path][0]
        if node.mount:
            mount = description.resolve_mount(path, node.mount)
            paths[path] = f"{paths[mount.entity]}/{mount.body}".rstrip("/")
        for reference in node.meshes:
            target = description.resolve_mount(path, Mount(reference)).entity
            if target not in keys:
                raise ValueError(f"Entity {path}: mesh must reference a concrete asset: {reference}")
    # Two independent entities must never accidentally author the same prim.
    occupied = {}
    for path, prim in paths.items():
        if not isinstance(index[path][0], Sensor):
            if prim in occupied:
                raise ValueError(f"Native prim collision: {path} and {occupied[prim]} at {prim}")
            occupied[prim] = path
    return SceneMetadata(keys, paths, scopes)
