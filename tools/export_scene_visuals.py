"""Export one native scene's initial visual meshes for a lightweight Blender editor.

Uses the installed SDK factories and terrain generator, on CPU, in a separate
process. Geometry is root-local; generated terrain is shifted by env0's origin.
No training, policy, Blender-side physics, or hand-recreated robot mesh is used.
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
    from shared.assembly.catalog import _project_package
    from shared.assembly.interfaces import source_revision
    from shared.assembly.project import prepare_geometry
    revision = source_revision(scene)
    return prepare_geometry(scene, project_package=_project_package(scene.parent)), revision


def export(scene, *, force=False):
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    prepared, revision = _prepare(scene)
    from shared.backends.isaac.environment import launch_application
    from shared.isaac_state import ProcessExit
    owner = ProcessExit(launch_application(prepared.application))
    try:
        _export(scene, prepared, revision, force=force)
    finally:
        # This exporter owns a disposable process. Stopping a standalone Kit
        # timeline can block headless shutdown; ProcessExit flushes and exits.
        owner.close()


def _export(scene, prepared, revision, *, force=False):
    """One geometry snapshot inside a running Kit application, on a fresh stage."""
    from shared.assembly.interfaces import source_revision
    from shared.assembly.visuals import cache_path, geometry_keys, load_visuals
    from shared.assembly.mesh_preview import MESH_SOURCE_SUFFIX, write_mesh_source
    from shared.backends.isaac.scene import scene_metadata, reset_authored_assets
    keys = geometry_keys(scene)
    existing = {} if force else load_visuals(scene)
    simulation = scene_instance = None
    try:
        from dataclasses import replace
        from isaaclab.scene import InteractiveScene
        from isaaclab.assets import AssetBaseCfg, ArticulationCfg
        from isaaclab.sim import SimulationContext, SimulationCfg
        from isaaclab.sim.utils.stage import attach_stage_to_usd_context, create_new_stage, use_stage
        from shared.assembly.spec import Sensor
        from shared.backends.isaac.scene import build_scene_config
        from isaaclab.utils.seed import configure_seed
        from types import SimpleNamespace
        def geometry_only(node, parent=''):
            path = parent + '/' + node.name if parent else node.name
            if isinstance(node, Sensor):
                return None
            children = tuple(child for item in node.children
                             if (child := geometry_only(item, path)) is not None)
            if node.factory and path in existing:
                # Keep only hierarchy transforms for any uncached descendants.
                if not children:
                    return None
                return replace(node, factory=None, parameters={}, initial_state={}, frames=(),
                               connections=(), reset=None, children=children)
            return replace(node, connections=(), reset=None, children=children)
        # Root-local cached assets need not be spawned again. No tasks, cameras,
        # observations, material imports in Blender, or training managers.
        prepared.description = replace(prepared.description, root=geometry_only(prepared.description.root), connections=())
        from pxr import Usd, UsdGeom, UsdPhysics, Gf
        import numpy as np
        configure_seed(42)
        scene_cfg = build_scene_config(prepared.description, factories=prepared.factories,
                                       num_envs=1, env_spacing=4, replicate_physics=False,
                                       clone_in_fabric=False)
        # Export independent root-local assets, without their assembly joints.
        # Pin articulation roots for this snapshot: disabling a USD world joint
        # without the assembly's root relocation can leave PhysX no valid root.
        # This touches only temporary preview configs, never training or YAML.
        for config in vars(scene_cfg).values():
            if isinstance(config, ArticulationCfg):
                props = getattr(config.spawn, 'articulation_props', None)
                if props is not None:
                    props.fix_root_link = True
        # A long-lived exporter serves several snapshots; each starts from an empty stage.
        create_new_stage()
        simulation = SimulationContext(SimulationCfg(device='cpu', use_fabric=False))
        with use_stage(simulation.get_initial_stage()):
            scene_instance = InteractiveScene(scene_cfg)
            attach_stage_to_usd_context()
        native = SimpleNamespace(scene=scene_instance, sim=simulation, num_envs=1, device='cpu',
                                 cfg=SimpleNamespace(scene=scene_cfg))
        with use_stage(simulation.get_initial_stage()):
            simulation.reset()
        metadata = scene_metadata(prepared.description)
        names = [key for path, key in metadata.keys.items() if metadata.scopes[path] == 'environment'
                 and isinstance(getattr(scene_cfg, key), AssetBaseCfg)]
        reset_authored_assets(native, None, asset_names=names)
        native.scene.write_data_to_sim()
        native.sim.physics_sim_view.update_articulations_kinematic()
        native.sim.forward()
        stage = native.scene.stage
        transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
        metadata = scene_metadata(prepared.description)
        origin = native.scene.env_origins[0].cpu().tolist()
        output = cache_path(scene)
        entities = dict(existing)
        for path, (node, _) in prepared.description.index().items():
            if not node.factory:
                continue
            key = metadata.keys[path]
            record = {'key': keys[path], 'path': None, 'exact': True,
                      'note': 'Native initial visual geometry; materials and physics are not edited in Blender'}
            factory_cfg = prepared.factories[node.factory](**node.parameters)
            initial = getattr(factory_cfg, 'init_state', None)
            if initial is not None:
                record['default_pose'] = {'position': list(initial.pos), 'orientation': list(initial.rot)}
            if key in native.scene.articulations or key in native.scene.rigid_objects:
                asset = native.scene[key]
                record['initial_state'] = {
                    'basis': 'Native asset defaults before task reset randomization',
                    'linear_velocity': asset.data.default_root_state[0, 7:10].cpu().tolist(),
                    'angular_velocity': asset.data.default_root_state[0, 10:13].cpu().tolist(),
                }
                if key in native.scene.articulations:
                    record['initial_state']['body_names'] = list(asset.body_names)
                    record['initial_state']['root_body'] = asset.body_names[0]
                    record['initial_state']['joint_positions'] = dict(zip(
                        asset.joint_names, asset.data.default_joint_pos[0].cpu().tolist()))
                    record['initial_state']['joint_velocities'] = dict(zip(
                        asset.joint_names, asset.data.default_joint_vel[0].cpu().tolist()))
            if node.kind == 'sensor' or 'light' in node.factory or node.factory == 'clear_sky':
                entities[path] = record
                continue
            prim_path = metadata.prim_paths[path].replace('{ENV_REGEX_NS}', '/World/envs/env_0')
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
            # USD may retain the source asset's rest pose even after a tensor
            # joint reset. Use live PhysX FK for each articulation link, while
            # keeping that link's original visual mesh transform and scale.
            body_transforms = {}
            def matrix_from_pose(position, quaternion):
                transform = Gf.Transform()
                transform.SetTranslation(Gf.Vec3d(*[float(v) for v in position]))
                transform.SetRotation(Gf.Rotation(Gf.Quatd(float(quaternion[0]), Gf.Vec3d(*[float(v) for v in quaternion[1:]]))))
                return transform.GetMatrix()
            if key in native.scene.articulations:
                asset = native.scene.articulations[key]
                inverse = matrix_from_pose(asset.data.root_pos_w[0], asset.data.root_quat_w[0]).GetInverse()
                links = asset.root_physx_view.get_link_transforms()[0].cpu()
                for index, name in enumerate(asset.body_names):
                    body_prim = stage.GetPrimAtPath(prim_path + '/' + name)
                    if body_prim:
                        pose = links[index]
                        live = matrix_from_pose(pose[:3], pose[[6, 3, 4, 5]])
                        body_transforms[str(body_prim.GetPath())] = (transforms.GetLocalToWorldTransform(body_prim).GetInverse(), live)
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
            if key in native.scene.articulations:
                asset = native.scene.articulations[key]
                for body in asset.body_names:
                    body_prim = stage.GetPrimAtPath(prim_path + '/' + body)
                    if body_prim:
                        transform = Gf.Transform(body_transforms[str(body_prim.GetPath())][1] * inverse)
                        quat = transform.GetRotation().GetQuat()
                        frames[body] = {'position': list(transform.GetTranslation()),
                                        'orientation': [quat.GetReal(), *quat.GetImaginary()]}
            joint_frames = []
            if key in native.scene.articulations:
                asset = native.scene.articulations[key]
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
            raise ValueError('Project changed during geometry export; reload and retry')
        result = {'seed': 42, 'num_envs': 1, 'environment_origin': origin, 'entities': entities,
                  'exported_entities': [path for path, (node, _) in prepared.description.index().items() if node.factory],
                  'reused_entities': sorted(existing), 'pipeline': 'geometry-only'}
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2)+'\n')
        temp.replace(output)
        print(json.dumps({k:{q:v for q,v in r.items() if q not in ('key',)} for k,r in entities.items()}))
    finally:
        if scene_instance is not None:
            for group in ('articulations', 'rigid_objects', 'rigid_object_collections', 'deformable_objects', 'extras'):
                for asset in getattr(scene_instance, group, {}).values():
                    if hasattr(asset, '_clear_callbacks'):
                        asset._clear_callbacks()
        if simulation is not None:
            simulation.clear_all_callbacks()
            SimulationContext.clear_instance()


class _Restart(Exception):
    """The running Kit cannot serve this request (other launch options or changed code)."""


def _loaded_sources():
    sources = {}
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None) or ''
        if file.startswith(str(ROOT)) and os.path.isfile(file):
            sources[file] = os.stat(file).st_mtime_ns
    return sources


def serve(reply, idle):
    """Keep Kit running for the web editor: one JSON request per stdin line, one reply line.

    A request is ``{"scene": path, "force": bool}``; Kit starts with the first export.

    Exits on stdin EOF (editor gone), after ``idle`` seconds without requests, after
    any failed export, or when workspace code it imported has changed on disk.
    """
    import select
    import traceback
    os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
    from shared.assembly.visuals import ensure_visuals
    owner, options, sources = None, None, {}

    def start(prepared):
        nonlocal owner, options
        if owner is None:
            from shared.backends.isaac.environment import launch_application
            from shared.isaac_state import ProcessExit
            options = dict(prepared.application)
            owner = ProcessExit(launch_application(prepared.application))
        elif prepared.application != options:
            raise _Restart('application options changed')

    def exporter(scene, force):
        prepared, revision = _prepare(scene)
        start(prepared)
        _export(scene, prepared, revision, force=force)

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
