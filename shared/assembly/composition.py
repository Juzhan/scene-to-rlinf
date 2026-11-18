"""What a static scene USD (an indoor or outdoor environment) is made of: its objects, as a tree.

An environment stays one ``usd_object`` entity (the simulator spawns it once); editors list and edit
its parts through this tree. Objects are found from the USD structure alone:

- below the default prim, ``Scope`` and untyped prims are containers (GRScenes ``Meshes/Base/wall``);
- the first ``Xform`` or gprim on a path is an object (a wall, a cabinet model, a shelf, a house), unless
  it groups more than ``GROUP_CHILDREN`` geometric children of its own (Isaac's ``Root`` with 3 000
  meshes), or several holding more than ``GROUP_GPRIMS`` gprims (Rivermark's ``main``: props, foliage,
  roads), in which case it is a container too;
- lights are objects without geometry; materials, physics materials and guide prims (the colliders the
  environment wrapper adds) are skipped.

Paths are relative to the default prim (``Meshes/Base/wall/wall_3``): ``usd_object.exclude`` lists such
paths to remove (``asset_formats.pruned_layer``), and the editor's lightweight mesh has one part per
object, named by its path, so each can be selected, hidden or removed (``display_parts``: up to
``MAX_PARTS``; past that the folders with most objects, walls and floor tiles first in a house, are drawn
as one part each, their objects still listed).
"""
from functools import lru_cache
import os
from pathlib import Path
import re

GROUP_CHILDREN = 8          # an Xform with more geometric children than this is a container
GROUP_GPRIMS = 1000         # ... and so is one of several geometric children holding more gprims than this
FLAT = 40                   # containers with more objects than this group them by name stem in editors
MAX_PARTS = 150             # editor mesh parts per scene: the browser keeps one scene node per part
CACHE = Path(__file__).resolve().parents[2] / '.cache/composition'
REST = '其他'                # name group of the one-off names left in a long flat folder
SKIPPED_NAMES = {'Looks', 'Materials', 'PhysicsMaterial', 'PhysicsMaterials', '_materials'}


def _kind(prim, gprims):
    """'object', 'light', 'container' or 'skip'; ``gprims`` counts the gprims below each prim path."""
    from pxr import UsdGeom, UsdLux, UsdShade
    if prim.IsA(UsdShade.Material) or prim.IsA(UsdShade.Shader) or prim.GetName() in SKIPPED_NAMES:
        return 'skip'
    if prim.IsA(UsdGeom.Imageable) and UsdGeom.Imageable(prim).GetPurposeAttr().Get() == UsdGeom.Tokens.guide:
        return 'skip'  # simplified or footprint colliders written by the environment wrapper
    if prim.HasAPI(UsdLux.LightAPI) or prim.GetTypeName().endswith('Light'):
        return 'light'
    if prim.IsA(UsdGeom.Gprim):
        return 'object'
    if prim.IsA(UsdGeom.Xformable) and prim.GetTypeName() not in ('Scope', ''):
        geometric = [child for child in prim.GetFilteredChildren(_traversal()) if gprims.get(child.GetPath())]
        grouped = len(geometric) > GROUP_CHILDREN or (len(geometric) > 1 and gprims[prim.GetPath()] > GROUP_GPRIMS)
        return 'container' if grouped else 'object'
    return 'container'


def _gprim_counts(top):
    """{prim path: gprims at or below it} for ``top`` and everything under it, in one traversal."""
    from pxr import Usd, UsdGeom
    counts, root = {}, top.GetPath()
    for prim in Usd.PrimRange(top, _traversal()):
        if prim.IsA(UsdGeom.Gprim):
            path = prim.GetPath()
            while True:
                counts[path] = counts.get(path, 0) + 1
                if path == root:
                    break
                path = path.GetParentPath()
    return counts


def objects_below(top):
    """``[(relative path, kind)]`` of the objects under ``top`` (a USD prim), in stage order; kind is
    ``object`` (has geometry) or ``light``."""
    found, gprims = [], _gprim_counts(top)

    def walk(prim, prefix):
        for child in prim.GetFilteredChildren(_traversal()):
            kind = _kind(child, gprims)
            path = f'{prefix}/{child.GetName()}' if prefix else child.GetName()
            if kind == 'container':
                walk(child, path)
            elif kind == 'light' or (kind == 'object' and gprims.get(child.GetPath())):
                found.append((path, kind))
    walk(top, '')
    return found


def _traversal():
    from pxr import Usd
    return Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)


def scene_objects(path):
    """Objects of the USD file's default prim (cached per file version, in memory and in
    ``.cache/composition``: a large outdoor scene takes 15 s to traverse); () when it has none."""
    path = Path(path).resolve()
    return _scene_objects(str(path), path.stat().st_mtime_ns)


def cached_scene_objects(path):
    """``scene_objects`` when its analysis is cached on disk, else None (without opening the file)."""
    import json
    path = Path(path).resolve()
    cached = _cache_file(str(path), path.stat().st_mtime_ns)
    return tuple(tuple(item) for item in json.loads(cached.read_text())) if cached.is_file() else None


def _cache_file(path, mtime):
    import hashlib
    rules = Path(__file__).read_bytes()  # a change of these rules redoes the analysis
    return CACHE / (hashlib.sha256(f'{path}\0{mtime}'.encode() + rules).hexdigest()[:20] + '.json')


@lru_cache(maxsize=32)
def _scene_objects(path, mtime):
    import json
    from pxr import Usd
    cached = _cache_file(path, mtime)
    if cached.is_file():
        return tuple(tuple(item) for item in json.loads(cached.read_text()))
    stage = Usd.Stage.Open(path)
    top = stage.GetDefaultPrim()
    found = tuple(objects_below(top)) if top else ()
    cached.parent.mkdir(parents=True, exist_ok=True)
    temporary = cached.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(found))
    os.replace(temporary, cached)
    return found


def is_composite(objects):
    """Whether an editor lists the parts of an entity: a scene of several objects."""
    return sum(kind == 'object' for _, kind in objects) >= 2


_STEM = re.compile(r'(_?\d+)+$')
_OPAQUE = re.compile(r'^(model_[0-9a-f]{16,}|N__AA__|L__AA__)')


def tree(objects, exclude=()):
    """Editor rows for ``objects``: ``[{'id', 'parent', 'label', 'kind', 'count'}]`` in display order.

    Container paths become folders; a folder with more than ``FLAT`` objects groups them by name stem
    (``SM_floor27`` ... ``SM_floor99`` -> ``SM_floor ×73``; one-off names beside such groups -> ``其他 ×N``). Opaque generated names (GRScenes hashes,
    encoded light names) are shown as the folder name and an index; a folder of one object is that object."""
    excluded = set(exclude)
    rows, folders = [], {}

    def folder(path):
        if path in folders or not path:
            return path or None
        parent = folder(path.rpartition('/')[0])
        folders[path] = {'id': path, 'parent': parent, 'label': path.rpartition('/')[2], 'kind': 'folder', 'count': 0}
        rows.append(folders[path])
        return path
    by_parent = {}
    for path, kind in objects:
        by_parent.setdefault(path.rpartition('/')[0], []).append((path, kind))
    for parent_path, members in by_parent.items():
        parent = folder(parent_path)
        groups = {}
        if len(members) > FLAT:
            for path, kind in members:
                name = path.rpartition('/')[2]
                if not _OPAQUE.match(name):  # generated names share no meaningful stem
                    groups.setdefault(_STEM.sub('', name) or name, []).append(path)
            groups = {stem: paths for stem, paths in groups.items() if len(paths) > 2}
            grouped = {path for paths in groups.values() for path in paths}
            rest = [path for path, _ in members if path not in grouped]
            if groups and len(rest) > FLAT:  # the one-off names of a long flat list (Isaac's office)
                groups[REST] = rest
        group_of = {path: stem for stem, paths in groups.items() for path in paths}
        index = 0
        for path, kind in members:
            name = path.rpartition('/')[2]
            owner = parent
            stem = group_of.get(path)
            if stem is not None:
                group_id = f'{parent_path}/{stem} ×' if parent_path else f'{stem} ×'
                if group_id not in folders:
                    folders[group_id] = {'id': group_id, 'parent': parent, 'label': f'{stem} ×{len(groups[stem])}',
                                         'kind': 'folder', 'count': 0}
                    rows.append(folders[group_id])
                owner = group_id
            index += 1
            label = f'{(parent_path.rpartition("/")[2] or kind)} {index}' if _OPAQUE.match(name) or len(name) > 40 else name
            rows.append({'id': path, 'parent': owner, 'label': label, 'kind': kind, 'count': 1,
                         'excluded': path in excluded or any(path.startswith(e + '/') for e in excluded)})
    for row in rows:  # objects per folder, for the folder's label
        if row['kind'] != 'folder':
            parent = row['parent']
            while parent is not None:
                folders[parent]['count'] += 1
                parent = folders[parent]['parent']
    # A folder holding a single object (GRScenes ``table/model_<hash>_0``) is shown as that object, by its name.
    only = {}
    for row in rows:
        if row['parent'] is not None:
            only.setdefault(row['parent'], []).append(row)
    single = {}
    for folder_id, children in only.items():
        if len(children) == 1 and children[0]['kind'] != 'folder' and not folder_id.endswith(' ×'):
            child, row = children[0], folders[folder_id]
            child.update(parent=row['parent'], label=row['label'])
            single[folder_id] = child
    return [row for row in rows if row['id'] not in single or row['kind'] != 'folder']


def object_of(name, objects):
    """The object (relative path) a mesh prim path relative to the same top belongs to, or None."""
    paths = {path for path, _ in objects}
    while name:
        if name in paths:
            return name
        name = name.rpartition('/')[0]
    return None


def members(rows, row_id):
    """Object paths under a tree row (``tree``): the object itself, or every object of a folder."""
    children = {}
    for row in rows:
        children.setdefault(row['parent'], []).append(row)
    found, stack = [], [row for row in rows if row['id'] == row_id]
    while stack:
        row = stack.pop()
        if row['kind'] == 'folder':
            stack.extend(children.get(row['id'], ()))
        else:
            found.append(row['id'])
    return found


def display_parts(objects, cap=MAX_PARTS):
    """``{part: [object paths]}`` of the editor's mesh: one part per object, named by its path; in a scene of
    more than ``cap`` objects (a GRScenes shop: 7 000 goods in one folder), the folders with the most
    objects of their own are drawn as one part each, named by their ``tree`` row, until ``cap`` remain."""
    rows = tree(objects)
    parts = {row['id']: [row['id']] for row in rows if row['kind'] == 'object'}
    if len(parts) <= cap:
        return parts
    direct = {}
    for row in rows:
        if row['kind'] == 'object' and row['parent'] is not None:
            direct.setdefault(row['parent'], []).append(row['id'])
    for folder, names in sorted(direct.items(), key=lambda item: -len(item[1])):
        if len(parts) <= cap or len(names) < 2:
            break
        for name in names:
            del parts[name]
        parts[folder] = names
    return parts


def model_key(prim):
    """What model a scene object instantiates: the first asset (and prim) referenced at or below it, or
    its instance prototype; None for geometry authored in place. Objects of one key are copies (Isaac's
    office: 3 202 objects of 500 models; Rivermark: 1 038 identical road reflectors), which editors can
    simplify once and draw as instances."""
    from pxr import Usd
    for part in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        for spec in part.GetPrimStack():
            if spec.hasReferences:
                for reference in spec.referenceList.GetAddedOrExplicitItems():
                    if reference.assetPath:
                        layer = spec.layer
                        return layer.ComputeAbsolutePath(reference.assetPath), str(reference.primPath)
        if part.IsInstance():
            return 'prototype', str(part.GetPrototype().GetPath())
    return None


if __name__ == '__main__':  # analyse scene files ahead (the editor runs this in a process of its own)
    import sys
    for file in sys.argv[1:]:
        print(file, sum(kind == 'object' for _, kind in scene_objects(file)), flush=True)
