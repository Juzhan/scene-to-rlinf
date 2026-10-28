"""Export one native scene's initial visual meshes for the scene editors.

The project's training backend (``train.yaml``) builds the scene with its own SDK
factories and terrain generator through its ``visuals`` hook, in a separate process
running that backend's interpreter; this script extracts the USD meshes. Geometry is
root-local; generated terrain is shifted by env0's origin. No training, policy,
editor-side physics, or hand-recreated robot mesh is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _prepare(scene):
    """Static scene preparation; the revision is taken first so later edits are detected."""
    import run  # workspace/RLinf module paths before SDK integrations
    from shared.assembly.assets import backend_module
    from shared.assembly.catalog import _project_package
    from shared.assembly.interfaces import source_revision
    from shared.assembly.project import prepare_geometry
    from shared.assembly.visuals import geometry_backend
    revision = source_revision(scene)
    backend = geometry_backend(scene)
    prepared = prepare_geometry(scene, project_package=_project_package(scene.parent), backend=backend)
    return backend_module(backend, 'visuals'), prepared, revision


def export(scene, *, force=False):
    visuals, prepared, revision = _prepare(scene)
    owner = visuals.launch(prepared)
    try:
        _export(scene, visuals, prepared, revision, force=force)
    finally:
        owner.close()


def _matrix(position, quaternion):
    from pxr import Gf
    transform = Gf.Transform()
    transform.SetTranslation(Gf.Vec3d(*[float(v) for v in position]))
    transform.SetRotation(Gf.Rotation(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*[float(v) for v in quaternion[1:]]))))
    return transform.GetMatrix()


def geometry_only(description, cached):
    """The scene reduced to entities whose native geometry is not in ``cached`` (paths).

    Sensors and abstract entities have no native geometry; a mounted one would also
    reference an entity that is cached and therefore not spawned again.
    """
    from dataclasses import replace
    from shared.assembly.spec import Abstract, Sensor

    def keep(node, parent=''):
        path = parent + '/' + node.name if parent else node.name
        if isinstance(node, (Sensor, Abstract)):
            return None
        children = tuple(child for item in node.children if (child := keep(item, path)) is not None)
        if node.factory and path in cached:
            # Keep only hierarchy transforms for any uncached descendants.
            if not children:
                return None
            return replace(node, factory=None, parameters={}, initial_state={}, frames=(),
                           connections=(), reset=None, children=children)
        return replace(node, connections=(), reset=None, children=children)
    return replace(description, root=keep(description.root), connections=())


def _export(scene, visuals, prepared, revision, *, force=False):
    """One geometry snapshot in the backend's running SDK (``visuals``), on a fresh stage."""
    from shared.assembly.interfaces import source_revision
    from shared.assembly.visuals import SceneChanged, cache_path, geometry_keys, load_visuals
    from shared.assembly.mesh_preview import MESH_SOURCE_SUFFIX, write_mesh_source
    keys = geometry_keys(scene)
    existing = {} if force else load_visuals(scene)
    # Root-local cached assets need not be spawned again. No tasks, cameras, observations,
    # material imports, training managers or assembly joints: each asset is root-local.
    description = prepared.description
    prepared.description = geometry_only(description, existing)
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    import numpy as np
    with visuals.snapshot(prepared) as native:
        stage, origin = native.stage, native.origin
        transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        output = cache_path(scene)
        output.parent.mkdir(parents=True, exist_ok=True)
        entities = dict(existing)
        for path, (node, _) in prepared.description.index().items():
            if not node.factory:
                continue
            state = native.entity(path, node)
            record = {'key': keys[path], 'path': None, 'exact': True,
                      'note': 'Native initial visual geometry; materials and physics are not edited in the editor'}
            for name in ('default_pose', 'initial_state'):
                if name in state:
                    record[name] = state[name]
            if node.kind == 'sensor' or 'light' in node.factory or node.factory == 'clear_sky':
                entities[path] = record
                continue
            prim_path = state['prim_path']
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                entities[path] = record
                continue
            root_transform = transforms.GetLocalToWorldTransform(prim)
            if node.factory == 'terrain':
                root_transform = Gf.Matrix4d(1).SetTranslate(Gf.Vec3d(*origin))
                record['note'] = 'Native generated terrain, seed42/env0; coordinates relative to the selected terrain origin'
                record['environment_origin'] = origin
            inverse = root_transform.GetInverse()
            # Each articulation link uses the backend's live FK pose while keeping that
            # link's original visual mesh transform and scale.
            body_transforms = {}
            if 'links' in state:
                inverse = _matrix(*state['root_pose']).GetInverse()
                for name, pose in state['links'].items():
                    body_prim = stage.GetPrimAtPath(prim_path + '/' + name)
                    if body_prim:
                        body_transforms[str(body_prim.GetPath())] = (
                            transforms.GetLocalToWorldTransform(body_prim).GetInverse(), _matrix(*pose))
            parts = []
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if not child.IsA(UsdGeom.Mesh):
                    continue
                mesh = UsdGeom.Mesh(child)
                if mesh.ComputeVisibility() == 'invisible' or mesh.ComputePurpose() in ('guide', 'proxy'):
                    continue
                points = mesh.GetPointsAttr().Get()
                counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
                if points is None or counts is None or not len(points):
                    continue
                world = transforms.GetLocalToWorldTransform(child)
                for body_path, (body_inverse, live) in body_transforms.items():
                    if str(child.GetPath()).startswith(body_path + '/'):
                        world = world * body_inverse * live
                        break
                matrix = np.asarray(world * inverse)
                vertices = np.asarray(points, dtype=np.float64) @ matrix[:3, :3] + matrix[3, :3]
                counts, indices = np.asarray(counts, dtype=np.int64), np.asarray(indices, dtype=np.int64)
                starts = np.cumsum(counts) - counts
                # Fan-triangulate every polygon at once; per-face Python loops dominate large CAD meshes.
                sizes = np.maximum(counts - 2, 0)
                first = np.repeat(starts, sizes)
                step = np.arange(sizes.sum()) - np.repeat(np.cumsum(sizes) - sizes, sizes) + 1
                faces = np.stack([indices[first], indices[first + step], indices[first + step + 1]], axis=1)
                if node.factory == 'terrain':
                    # Show one real training tile instead of surrounding 160
                    # curriculum copies; its centre is the project's local zero.
                    half = np.asarray(node.parameters['terrain_generator']['size']) * .5
                    filled = counts > 0
                    centres = np.add.reduceat(vertices[indices, :2], starts[filled]) / counts[filled, None]
                    inside = np.zeros(len(counts), dtype=bool)
                    inside[filled] = (np.abs(centres) <= half + 1e-5).all(axis=1)
                    faces = faces[np.repeat(inside, sizes)]
                    if not len(faces):
                        continue
                    used, faces = np.unique(faces, return_inverse=True)
                    vertices, faces = vertices[used], faces.reshape(-1, 3)
                    record['note'] = 'Native generated terrain tile, seed42/env0; local zero is this tile centre. Other training copies hidden.'
                parts.append({'name': str(child.GetPath()), 'vertices': vertices, 'faces': faces})
            frames = {}
            for body_path, (_, live) in body_transforms.items():
                transform = Gf.Transform(live * inverse)
                quat = transform.GetRotation().GetQuat()
                frames[body_path.rsplit('/', 1)[1]] = {'position': list(transform.GetTranslation()),
                                                       'orientation': [quat.GetReal(), *quat.GetImaginary()]}
            joint_frames = []
            if 'links' in state:
                positions = record['initial_state']['joint_positions']
                for joint_prim in Usd.PrimRange(prim):
                    if not joint_prim.IsA(UsdPhysics.Joint):
                        continue
                    joint = UsdPhysics.Joint(joint_prim)
                    parents, children = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                    if not parents or not children:
                        continue  # World/root anchoring is not an articulation DOF.
                    parent, child = parents[0].name, children[0].name
                    if parent not in frames or child not in frames:
                        continue
                    name = joint_prim.GetName()
                    pos, rot = joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get()
                    row = {'name': name, 'parent': parent, 'child': child,
                           'origin': {'position': list(pos), 'orientation': [rot.GetReal(), *rot.GetImaginary()]},
                           'position': positions.get(name, 0.), 'type': 'fixed', 'axis': 'X'}
                    for schema, kind, factor in ((UsdPhysics.RevoluteJoint, 'revolute', np.pi / 180.),
                                                 (UsdPhysics.PrismaticJoint, 'prismatic', 1.)):
                        if joint_prim.IsA(schema):
                            typed = schema(joint_prim)
                            row.update(type=kind, axis=str(typed.GetAxisAttr().Get()))
                            low, high = typed.GetLowerLimitAttr().Get(), typed.GetUpperLimitAttr().Get()
                            if np.isfinite(low) and np.isfinite(high) and low < high:
                                row['limits'] = [float(low * factor), float(high * factor)]
                    joint_frames.append(row)
            if parts:
                # Binary intermediate, read only by the preview simplifiers; the editors load the light mesh.
                filename = output.parent / (keys[path] + MESH_SOURCE_SUFFIX)
                temporary = filename.with_name(filename.name + '.tmp')
                with temporary.open('wb') as stream:
                    write_mesh_source(stream, parts, frames, joint_frames)
                temporary.replace(filename)
                record.update(path=str(filename), mesh_count=len(parts),
                              vertex_count=sum(len(p['vertices']) for p in parts))
            entities[path] = record
    for path, key in keys.items():
        entities.setdefault(path, {'key': key, 'path': None, 'exact': False,
                                  'note': 'Sensor helper; sensors are not started for geometry export'})
    if source_revision(scene) != revision:
        raise SceneChanged('Project changed during geometry export; reload and retry')
    result = {'seed': 42, 'num_envs': 1, 'environment_origin': origin, 'entities': entities,
              'exported_entities': [path for path, (node, _) in prepared.description.index().items() if node.factory],
              'reused_entities': sorted(existing), 'pipeline': 'geometry-only'}
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2)+'\n')
    temp.replace(output)
    print(json.dumps({k:{q:v for q,v in r.items() if q not in ('key',)} for k,r in entities.items()}))


class _Restart(Exception):
    """The running SDK cannot serve this request (other backend, launch options or changed code)."""


def _loaded_sources():
    sources = {}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None) or ''
        if file.startswith(str(ROOT)) and os.path.isfile(file):
            sources[file] = os.stat(file).st_mtime_ns
    return sources


def serve(reply, idle):
    """Keep the backend SDK (e.g. Kit) running for the web editor: one JSON request per stdin line, one reply line.

    A request is ``{"scene": path, "force": bool}``; the SDK starts with the first export.

    Exits on stdin EOF (editor gone), after ``idle`` seconds without requests, after
    any failed export, or when workspace code it imported has changed on disk.
    """
    import select
    import traceback
    from shared.assembly.visuals import SceneChanged, ensure_visuals
    owner, options, sources = None, None, {}

    def start(visuals, prepared):
        nonlocal owner, options
        current = (visuals.__name__, visuals.launch_options(prepared))
        if owner is None:
            options, owner = current, visuals.launch(prepared)
        elif current != options:
            raise _Restart('backend or launch options changed')

    def exporter(scene, force):
        visuals, prepared, revision = _prepare(scene)
        start(visuals, prepared)
        _export(scene, visuals, prepared, revision, force=force)

    try:
        while select.select([sys.stdin], [], [], idle)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            if any(not os.path.isfile(f) or os.stat(f).st_mtime_ns != t for f, t in sources.items()):
                reply.write(json.dumps({'restart': 'code changed'}) + '\n')
                break
            request = json.loads(line)
            try:
                result = ensure_visuals(Path(request['scene']), force=request.get('force', False), exporter=exporter)
            except _Restart as error:
                reply.write(json.dumps({'restart': str(error)}) + '\n')
                break
            except SceneChanged as error:
                # The snapshot finished cleanly on an outdated file; Kit stays usable.
                reply.write(json.dumps({'error': str(error), 'changed': True}) + '\n')
                continue
            except BaseException:
                # A failed snapshot may leave Kit in an unknown state: report and exit.
                reply.write(json.dumps({'error': traceback.format_exc()[-2500:]}) + '\n')
                break
            sources = {**_loaded_sources(), **sources}
            reply.write(json.dumps({'result': result}) + '\n')
    finally:
        # The editor may be gone (broken reply pipe); Kit must still be closed, otherwise
        # interpreter shutdown hangs with Kit alive and the process never releases the GPU.
        try:
            reply.flush()
        finally:
            if owner is not None:
                owner.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument('--serve', type=int, metavar='FD', help='Serve requests on stdin, replying on this file descriptor')
    parser.add_argument('--idle', type=float, default=120, help='Seconds a serving exporter waits for the next request')
    args = parser.parse_args()
    if args.serve is not None:
        serve(os.fdopen(args.serve, 'w', buffering=1), args.idle)
    elif args.scene:
        export(args.scene.resolve(), force=args.force)
    else:
        parser.error('--scene or --serve is required')
