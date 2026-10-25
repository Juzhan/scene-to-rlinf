"""Keep one robot/object asset available to both Isaac (USD) and MuJoCo/mjlab (MJCF).

Every conversion is done by an official converter; this module chains them through
URDF, stages meshes, and records provenance plus an alignment report in ASSET.json:

    USD  -> URDF : NVIDIA UsdToUrdf (Isaac Sim isaacsim.asset.exporter.urdf; pxr only)
    URDF -> MJCF : MuJoCo URDF compiler
    URDF -> USD  : Isaac Lab UrdfConverter (Isaac Sim URDF importer; starts Kit)
    MJCF -> USD  : Isaac Lab MjcfConverter (Isaac Sim MJCF importer; starts Kit)

The body must align (joints, limits, axes, masses, inertia, frames); appearance only
needs to be close. Links whose collision geometry covers their visuals (bounding boxes
within APPEARANCE_TOLERANCE) are drawn in MJCF from that geometry, coloured with the
dominant visual material colour. Elsewhere the visual meshes are kept and the collision
geoms move to the hidden group COLLISION_GROUP; meshes over MuJoCo's face limit are
reduced by vertex clustering. Actuators and gains are not part of the converted files:
each backend's asset factory declares them.

    run.py asset convert assets/piper_x/piper_x.usda assets/piper_x
    run.py asset check assets/piper_x

Upstream model files (USD, URDF or MJCF and everything they load) are copied unmodified,
at their repository or URL paths, into ASSET/source/, with SOURCE.json recording the origin
and the sha256 of each file:

    run.py asset vendor <checkout>/piper_description/urdf/piper_description.urdf assets/piper
    run.py asset vendor $NUCLEUS/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd assets/go2 --base $NUCLEUS

MuJoCo backends call mjcf_for(): the recorded conversion when present, else a cached one.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

ROOT = Path(__file__).resolve().parents[1]
USD_SUFFIXES = {'.usd', '.usda', '.usdc'}
MESH_SUFFIXES = {'.stl'}
TOLERANCE = {'position': 1e-4, 'mass': 1e-3, 'inertia': 1e-3, 'angle': 1e-4, 'axis': 1e-4}
PHYSX_DEFAULT_MASS = 1.0      # kg, PhysX's mass for a rigid body with neither a positive mass nor colliders
APPEARANCE_TOLERANCE = 0.005  # m, visual vs collision bounding box corners of one link
COLLISION_GROUP = 3           # MuJoCo viewers and mjlab cameras draw groups 0-2 by default
MESH_FACE_LIMIT = 200000      # MuJoCo mesh loader


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def workspace_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def source_format(path):
    suffix = Path(path).suffix.lower()
    if suffix in USD_SUFFIXES:
        return 'usd'
    if suffix == '.urdf':
        return 'urdf'
    if suffix == '.xml':
        return 'mjcf'
    raise ValueError(f'{path}: expected .usd/.usda/.usdc, .urdf or MJCF .xml')


# USD -> URDF ---------------------------------------------------------------

def _usd_to_urdf_library():
    spec = importlib.util.find_spec('isaacsim')
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError('USD -> URDF needs Isaac Sim (isaacsim.asset.exporter.urdf)')
    bundle = Path(next(iter(spec.submodule_search_locations))) / 'exts/isaacsim.asset.exporter.urdf/pip_prebundle'
    if not bundle.is_dir():
        raise RuntimeError(f'Isaac Sim URDF exporter not found: {bundle}')
    import nvidia  # CUDA wheels own the namespace; the exporter ships nvidia.srl beside it
    if str(bundle / 'nvidia') not in nvidia.__path__:
        nvidia.__path__.append(str(bundle / 'nvidia'))
    if str(bundle) not in sys.path:
        sys.path.append(str(bundle))
    from nvidia.srl.from_usd.to_urdf import UsdToUrdf
    return UsdToUrdf, (bundle.parents[0] / 'config/extension.toml')


FIXED_JOINT_RECONCILE = {'angle': 0.01, 'position': 0.001}  # rad, m


def physx_stage(path):
    """Open a USD with fixed-joint children at the pose PhysX simulates, plus the bodies moved.

    A PhysX articulation places each link from its parent through the joint frames, so the
    authored pose of a fixed joint's child is ignored. Small disagreements between the two
    (ANYmal-C RH_FOOT: 1.2 mrad) make UsdToUrdf reject the asset; the child is moved to the
    parent-implied pose in the in-memory session layer only. Larger gaps are left for the
    exporter to report, and other joint types are never touched (their gap is a joint angle)."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())
    moved = {}

    def local(joint, side):
        position = getattr(joint, f'GetLocalPos{side}Attr')().Get() or Gf.Vec3f()
        rotation = getattr(joint, f'GetLocalRot{side}Attr')().Get() or Gf.Quatf(1)
        return Gf.Matrix4d().SetTransform(Gf.Rotation(Gf.Quatd(rotation)), Gf.Vec3d(position))

    for _ in range(64):  # A moved body can shift joints further down the chain.
        cache, changed = UsdGeom.XformCache(), False
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.FixedJoint):
                continue
            joint = UsdPhysics.FixedJoint(prim)
            ends = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
            if not all(ends):
                continue
            parent, child = (stage.GetPrimAtPath(targets[0]) for targets in ends)
            via_parent = local(joint, 0) * cache.GetLocalToWorldTransform(parent).RemoveScaleShear()
            child_world = cache.GetLocalToWorldTransform(child)
            via_child = local(joint, 1) * child_world.RemoveScaleShear()
            turn = via_parent.ExtractRotationQuat() * via_child.ExtractRotationQuat().GetInverse()
            angle = 2 * math.acos(min(1., abs(turn.GetReal())))
            offset = (via_parent.ExtractTranslation() - via_child.ExtractTranslation()).GetLength()
            if (angle <= 1e-5 and offset <= 1e-6) or angle > FIXED_JOINT_RECONCILE['angle'] \
                    or offset > FIXED_JOINT_RECONCILE['position']:
                continue
            if not Gf.IsClose(child_world.RemoveScaleShear(), child_world, 1e-9):
                raise ValueError(f'{child.GetPath()}: scaled fixed-joint child cannot be re-posed')
            target = local(joint, 1).GetInverse() * via_parent
            parent_world = cache.GetParentToWorldTransform(child)
            UsdGeom.Xformable(child).MakeMatrixXform().Set(target * parent_world.GetInverse())
            entry = moved.setdefault(child.GetName(), {'joint': prim.GetName(), 'angle': 0., 'position': 0.})
            entry['angle'], entry['position'] = max(entry['angle'], angle), max(entry['position'], offset)
            changed = True
            break
        if not changed:
            return stage, moved
    raise ValueError(f'{path}: fixed-joint poses did not settle')


def usd_to_urdf(source, directory, name, root=None):
    """Export the USD kinematic tree; link names lose the exporter's '<defaultPrim>_' prefix."""
    UsdToUrdf, extension = _usd_to_urdf_library()
    stage, moved = physx_stage(source)
    prefix = stage.GetDefaultPrim().GetName() + '_' if stage.GetDefaultPrim() else None
    exporter = UsdToUrdf(stage, root=root, log_level='WARNING')
    path = Path(directory) / (name + '.urdf')
    exporter.save_to_file(urdf_output_path=path, mesh_dir=Path(directory) / 'meshes',
                          mesh_path_prefix='./', quiet=True)
    tree = ElementTree.parse(path)
    robot = tree.getroot()
    names = [link.get('name') for link in robot.iter('link')]
    rename = {}
    if prefix:
        for value in names:
            short = value.removeprefix(prefix)
            if short != value and short not in names and short not in rename.values():
                rename[value] = short
    # A body that is the articulation root without a joint above it is merged into the
    # default-prim node and named after it (M20 base_link -> 'M20'); give it its body name.
    from pxr import UsdPhysics
    bodies = {prim.GetName() for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    children = {joint.find('child').get('link') for joint in robot.iter('joint')}
    roots = [value for value in names if value not in children]
    unnamed = bodies - {rename.get(value, value) for value in names}
    if len(roots) == 1 and roots[0] not in bodies and len(unnamed) == 1 and (
            stage.GetDefaultPrim() and roots[0] == stage.GetDefaultPrim().GetName()):
        rename[roots[0]] = next(iter(unnamed))
    for link in robot.iter('link'):
        link.set('name', rename.get(link.get('name'), link.get('name')))
    for joint in robot.iter('joint'):
        for key in ('parent', 'child'):
            element = joint.find(key)
            element.set('link', rename.get(element.get('link'), element.get('link')))
    for mesh in robot.iter('mesh'):  # exporter writes './meshes/...' relative to the mesh directory parent
        mesh.set('filename', str((Path(directory) / mesh.get('filename')).resolve()))
    tree.write(path)
    version = next((line.split('=')[1].strip().strip('"') for line in extension.read_text().splitlines()
                    if line.startswith('version')), 'unknown')
    return path, {'tool': 'isaacsim.asset.exporter.urdf (nvidia.srl UsdToUrdf)', 'version': version,
                  'renamed_links': rename, 'fixed_joint_children_posed_by_parent': moved}


# URDF -> MJCF --------------------------------------------------------------

def find_package(name, start):
    """The nearest ancestor directory tree holding package.xml with this <name>."""
    for root in [start, *start.parents]:
        for candidate in (root / name, root):
            manifest = candidate / 'package.xml'
            if manifest.is_file() and ElementTree.parse(manifest).findtext('name') == name:
                return candidate
        for manifest in root.glob('*/package.xml'):
            if ElementTree.parse(manifest).findtext('name') == name:
                return manifest.parent
    raise FileNotFoundError(f'package://{name} not found; pass --package {name}=PATH')


def resolve_mesh(filename, source, packages):
    if filename.startswith('package://'):
        name, _, relative = filename.removeprefix('package://').partition('/')
        if name not in packages:
            packages[name] = find_package(name, source.parent)
        return packages[name] / relative
    if filename.startswith('file://'):
        return Path(filename.removeprefix('file://'))
    return (source.parent / filename).resolve()


def material_colour(path):
    """Diffuse colour of the OBJ material covering most faces (None if unknown)."""
    if path.suffix.lower() != '.obj':
        return None
    library, counts, current = {}, Counter(), None
    with path.open(errors='ignore') as lines:
        for line in lines:
            if line.startswith('f '):
                counts[current] += 1
            elif line.startswith('usemtl '):
                current = line.split(maxsplit=1)[1].strip()
            elif line.startswith('mtllib '):
                material, name = path.parent / line.split(maxsplit=1)[1].strip(), None
                if material.is_file():
                    for entry in material.read_text(errors='ignore').splitlines():
                        if entry.startswith('newmtl '):
                            name = entry.split(maxsplit=1)[1].strip()
                        elif entry.startswith('Kd ') and name is not None:
                            library[name] = [float(value) for value in entry.split()[1:4]]
    for name, _ in counts.most_common():
        if name in library:
            return [*library[name], 1.0]
    return None


def urdf_colour(visual, source, packages):
    colour = visual.find('material/color')
    if colour is not None:
        return [float(value) for value in colour.get('rgba').split()]
    mesh = visual.find('geometry/mesh')
    return material_colour(resolve_mesh(mesh.get('filename'), source, packages)) if mesh is not None else None


def _rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def element_bounds(element, source, packages, meshes):
    """Link-frame bounding box (min, max) of a URDF visual/collision element, None if unknown."""
    import numpy as np
    geometry = element.find('geometry')
    shape = geometry[0] if geometry is not None and len(geometry) else None
    if shape is None:
        return None
    if shape.tag == 'mesh':
        path = resolve_mesh(shape.get('filename'), source, packages)
        if str(path) not in meshes:
            import trimesh
            meshes[str(path)] = trimesh.load(path, force='mesh').vertices
        scale = [float(value) for value in shape.get('scale', '1 1 1').split()]
        points = np.asarray(meshes[str(path)]) * scale
    else:
        if shape.tag == 'box':
            half = np.array([float(value) for value in shape.get('size').split()]) / 2
        elif shape.tag == 'cylinder':
            radius = float(shape.get('radius'))
            half = np.array([radius, radius, float(shape.get('length')) / 2])
        elif shape.tag == 'sphere':
            half = np.full(3, float(shape.get('radius')))
        else:
            return None
        points = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * half
    origin = element.find('origin')
    xyz = [float(value) for value in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
    rpy = [float(value) for value in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
    points = points @ np.array(_rotation(rpy)).T + xyz
    return points.min(0), points.max(0)


def same_shape(left, right):
    """Same geometry and origin in two URDF visual/collision elements."""
    def shape(element):
        origin, geometry = element.find('origin'), element.find('geometry')
        return ({} if origin is None else dict(origin.attrib),
                [(child.tag, dict(child.attrib)) for child in (geometry if geometry is not None else ())])
    return shape(left) == shape(right)


def bounds_difference(visuals, collisions, source, packages, meshes):
    """Largest corner distance between the visual and collision bounding boxes of a link."""
    import numpy as np

    def union(elements):
        boxes = [element_bounds(element, source, packages, meshes) for element in elements]
        if not boxes or any(box is None for box in boxes):
            return None
        return np.min([box[0] for box in boxes], 0), np.max([box[1] for box in boxes], 0)

    visual, collision = union(visuals), union(collisions)
    if visual is None or collision is None:
        return None
    return float(max(np.abs(visual[0] - collision[0]).max(), np.abs(visual[1] - collision[1]).max()))


def cluster_vertices(mesh, cell):
    """Merge vertices within one grid cell; drops collapsed and repeated faces."""
    import numpy as np
    import trimesh
    _, first, inverse = np.unique(np.floor(mesh.vertices / cell).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    faces = inverse.reshape(-1)[mesh.faces]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    _, unique = np.unique(np.sort(faces, 1), axis=0, return_index=True)
    return trimesh.Trimesh(mesh.vertices[first], faces[np.sort(unique)], process=False)


def write_mesh(path, target):
    """STL copy of a mesh; returns (faces before, faces after, cell m) when it had to be reduced."""
    import trimesh
    mesh = trimesh.load(path, force='mesh')  # geometry only: MuJoCo meshes carry no material or texture
    if len(mesh.faces) <= MESH_FACE_LIMIT:
        if path.suffix.lower() in MESH_SUFFIXES:
            shutil.copyfile(path, target)
        else:
            mesh.export(target)
        return None
    cell, reduced = 1e-4, mesh
    while len(reduced.faces) > MESH_FACE_LIMIT:
        reduced = cluster_vertices(mesh, cell)
        cell *= 2
    reduced.export(target)
    return [len(mesh.faces), len(reduced.faces), cell / 2]


def urdf_to_mjcf(source, directory, name, packages=None, keep_visuals=False):
    """Compile URDF with MuJoCo into directory/<name>.xml with STL meshes in directory/assets."""
    import mujoco
    source, directory = Path(source).resolve(), Path(directory).resolve()
    packages = {key: Path(value).resolve() for key, value in (packages or {}).items()}
    tree = ElementTree.parse(source)
    robot = tree.getroot()
    if robot.tag != 'robot':
        raise ValueError(f'{source}: not a URDF robot')
    meshes = directory / 'assets'
    meshes.mkdir(parents=True)
    colours, dropped, kept, simplified, written, by_content, vertices = {}, [], {}, {}, {}, {}, {}
    for link in robot.iter('link'):
        visuals = link.findall('visual')
        collisions = [element for element in link.findall('collision') if element.find('geometry') is not None]
        if not visuals or not collisions:
            continue
        # UsdToUrdf also writes every collision element as a visual; such copies show nothing new.
        copies = [visual for visual in visuals if any(same_shape(visual, element) for element in collisions)]
        visuals = [visual for visual in visuals if visual not in copies] or copies
        colour = next(filter(None, (urdf_colour(visual, source, packages) for visual in visuals)), None)
        if colour is not None:
            colours[link.get('name')] = colour
        for visual in copies:
            if visual not in visuals:
                link.remove(visual)
        difference = None if keep_visuals else bounds_difference(visuals, collisions, source, packages, vertices)
        if difference is not None and difference <= APPEARANCE_TOLERANCE:
            # MJCF is not the rendering format: draw the link with its collision
            # geometry in the dominant visual colour.
            for visual in visuals:
                link.remove(visual)
            dropped.append(link.get('name'))
        else:
            # Collision geometry that does not cover the visual would draw the part
            # misplaced: keep the visual and hide the collision geoms instead.
            kept[link.get('name')] = None if difference is None else round(difference, 4)
    for mesh in robot.iter('mesh'):
        path = resolve_mesh(mesh.get('filename'), source, packages)
        if not path.is_file():
            raise FileNotFoundError(f'Mesh not found: {mesh.get("filename")} -> {path}')
        key = str(path)
        if key not in written:
            content = digest(path)
            if content in by_content:
                written[key] = by_content[content]
            else:
                target = meshes / (path.stem + '.stl')
                if target.exists():
                    target = meshes / f'{path.stem}_{content[:8]}.stl'
                reduced = write_mesh(path, target)
                written[key] = by_content[content] = target.relative_to(directory).as_posix()
                if reduced:
                    simplified[written[key]] = reduced
        mesh.set('filename', written[key])
    # MuJoCo URDF extension: keep fixed-joint links (camera/tcp mounts) as bodies.
    extension = robot.find('mujoco')
    if extension is None:
        extension = ElementTree.SubElement(robot, 'mujoco')
    compiler = extension.find('compiler')
    if compiler is None:
        compiler = ElementTree.SubElement(extension, 'compiler')
    compiler.attrib.update(meshdir='.', fusestatic='false', discardvisual='false',
                           balanceinertia='true', strippath='false')
    staged = directory / (name + '.urdf')
    tree.write(staged)
    try:
        spec = mujoco.MjSpec.from_file(str(staged))
    finally:
        staged.unlink()
    spec.modelname = name
    for body in list(spec.worldbody.bodies):
        # A URDF 'world' link (UsdToUrdf exports the articulation's parent prim as one)
        # compiles to an empty body beside the real root; it carries nothing.
        if not (body.bodies or body.geoms or body.joints or body.sites or body.cameras or body.mass):
            spec.delete(body)
    for body in {*colours, *kept}:
        for geom in spec.body(body).geoms:
            collides = bool(geom.contype or geom.conaffinity)
            if body in kept and collides:
                geom.group = COLLISION_GROUP
            elif body in colours and (body in kept) != collides:
                geom.rgba = colours[body]
    spec.compile()
    (directory / (name + '.xml')).write_text(spec.to_xml())
    return directory / (name + '.xml'), {
        'tool': 'MuJoCo URDF compiler', 'version': mujoco.__version__,
        'packages': {key: workspace_path(value) for key, value in packages.items()},
        'visuals_replaced_by_collision_geometry': sorted(dropped),
        'visuals_kept': kept,
        'meshes_simplified': simplified,
    }


def authored_inertial(mass):
    """UsdPhysics leaves centre of mass and inertia to be computed from colliders unless authored."""
    centre, inertia = mass.GetCenterOfMassAttr(), mass.GetDiagonalInertiaAttr()
    return (centre.HasAuthoredValue() and all(math.isfinite(value) for value in centre.Get())
            and inertia.HasAuthoredValue() and any(inertia.Get()))


def usd_frames_into_mjcf(mjcf, usd):
    """Turn each matching MJCF body frame back to the USD rigid-body orientation.

    UsdToUrdf can turn a link frame (Piper X gripper_link2 by pi); mounts are authored in
    the USD body frame, so the body is re-expressed: contents and children keep their world
    pose, only the frame they are written in changes. Positions are left to compare().
    """
    import mujoco
    import numpy as np
    from pxr import UsdGeom, UsdPhysics
    stage = physx_stage(usd)[0]
    cache = UsdGeom.XformCache()
    spec = mujoco.MjSpec.from_file(str(mjcf))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)

    def rotate(turn, vector):
        result = np.zeros(3)
        mujoco.mju_rotVecQuat(result, np.asarray(vector, float), turn)
        return result

    def multiply(left, right):
        result = np.zeros(4)
        mujoco.mju_mulQuat(result, np.asarray(left, float), np.asarray(right, float))
        return result

    turned = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI) or spec.body(prim.GetName()) is None:
            continue
        rotation = cache.GetLocalToWorldTransform(prim).RemoveScaleShear().ExtractRotationQuat()
        target = np.array([rotation.GetReal(), *rotation.GetImaginary()])
        inverse, turn = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[model.body(prim.GetName()).id])
        mujoco.mju_mulQuat(turn, inverse, target)  # old body frame -> USD frame
        if 2 * math.acos(min(1., abs(turn[0]))) <= TOLERANCE['angle']:
            continue
        body = spec.body(prim.GetName())
        back = np.zeros(4)
        mujoco.mju_negQuat(back, turn)
        elements = (*body.geoms, *body.sites, *body.cameras, *body.bodies)
        if any(item.alt.type != mujoco.mjtOrientation.mjORIENTATION_QUAT for item in (body, *elements)) or any(
                not math.isnan(geom.fromto[0]) for geom in body.geoms) or body.lights or body.frames:
            raise NotImplementedError(f'{prim.GetName()}: only quaternion-posed geoms/sites/cameras/bodies '
                                      'can be re-expressed')
        body.quat = multiply(body.quat, turn)
        for item in elements:
            item.pos, item.quat = rotate(back, item.pos), multiply(back, item.quat)
        for joint in body.joints:
            joint.pos, joint.axis = rotate(back, joint.pos), rotate(back, joint.axis)
        if body.explicitinertial:
            body.ipos, body.iquat = rotate(back, body.ipos), multiply(back, body.iquat)
        turned.append(prim.GetName())
    spec.compile()
    Path(mjcf).write_text(spec.to_xml())
    return turned


COLLIDER_TYPES = {'Cube': 'box', 'Sphere': 'sphere', 'Cylinder': 'cylinder', 'Capsule': 'capsule', 'Mesh': 'mesh'}
CAPSULE_AXES = {'X': (math.sqrt(.5), 0., math.sqrt(.5), 0.), 'Y': (math.sqrt(.5), -math.sqrt(.5), 0., 0.),
                'Z': (1., 0., 0., 0.)}  # MuJoCo capsules lie along their local z


def usd_colliders(stage):
    """(collider gprim, owning rigid body or None); CollisionAPI may sit on an ancestor Xform."""
    from pxr import Usd, UsdGeom, UsdPhysics
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        body, collider, node = None, None, prim
        while node and not node.IsPseudoRoot():
            if collider is None and node.HasAPI(UsdPhysics.CollisionAPI):
                collider = UsdPhysics.CollisionAPI(node).GetCollisionEnabledAttr().Get() is not False
            if node.HasAPI(UsdPhysics.RigidBodyAPI):
                body = node
                break
            node = node.GetParent()
        if collider:
            yield prim, body


def usd_capsules_into_mjcf(mjcf, usd):
    """Add the USD capsule colliders that URDF cannot express (UsdToUrdf skips them).

    Each capsule keeps its USD world pose, written in the matching MJCF body frame. Bodies
    that draw visual meshes get it in the hidden COLLISION_GROUP like their other colliders.
    """
    import mujoco
    import numpy as np
    from pxr import Gf, UsdGeom
    stage = physx_stage(usd)[0]
    cache = UsdGeom.XformCache()
    spec = mujoco.MjSpec.from_file(str(mjcf))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    added = Counter()
    for prim, owner in usd_colliders(stage):
        if not prim.IsA(UsdGeom.Capsule):
            continue
        body = spec.body(owner.GetName()) if owner else None
        if body is None:
            raise ValueError(f'{prim.GetPath()}: capsule collider without a matching MJCF body')
        capsule = UsdGeom.Capsule(prim)
        axis = capsule.GetAxisAttr().Get()
        transform = cache.GetLocalToWorldTransform(prim)
        scale = Gf.Transform(transform).GetScale()
        along = 'XYZ'.index(axis)
        radial = [scale[index] for index in range(3) if index != along]
        if abs(radial[0] - radial[1]) > 1e-6 * max(radial):
            raise ValueError(f'{prim.GetPath()}: capsule radius scaled unevenly {tuple(scale)}')
        rotation = transform.RemoveScaleShear().ExtractRotationQuat()
        world = np.zeros(4)
        mujoco.mju_mulQuat(world, np.array([rotation.GetReal(), *rotation.GetImaginary()]),
                           np.array(CAPSULE_AXES[axis]))
        index = model.body(owner.GetName()).id
        inverse, local = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[index])
        mujoco.mju_mulQuat(local, inverse, world)
        hidden = any(not (geom.contype or geom.conaffinity) for geom in body.geoms)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      size=[capsule.GetRadiusAttr().Get() * radial[0], capsule.GetHeightAttr().Get() / 2 * scale[along], 0],
                      pos=data.xmat[index].reshape(3, 3).T @ (np.array(transform.ExtractTranslation()) - data.xpos[index]),
                      quat=local, group=COLLISION_GROUP if hidden else 0)
        added[owner.GetName()] += 1
    spec.compile()
    Path(mjcf).write_text(spec.to_xml())
    return dict(sorted(added.items()))


def usd_inertia_into_mjcf(mjcf, usd):
    """Set each matching body's inertial from the USD MassAPI, expressed in the MJCF body frame.

    UsdToUrdf moves a link frame onto its joint frame when the joint's body-1 pose is not
    identity but copies centre of mass and principal axes unchanged, so they end up in the
    wrong frame (Piper X gripper_link2). The USD MassAPI stays the source of truth.
    With only a mass authored, MuJoCo derives the inertial from the collision geometry
    at that mass, as PhysX does from the colliders.
    """
    import mujoco
    import numpy as np
    from pxr import Gf, UsdGeom, UsdPhysics
    stage = physx_stage(usd)[0]
    cache = UsdGeom.XformCache()
    spec = mujoco.MjSpec.from_file(str(mjcf))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    from_geometry = spec.copy()
    from_geometry.compiler.inertiafromgeom = mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    geometric = from_geometry.compile()
    updated = []
    for prim in stage.Traverse():
        if not (prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI)):
            continue
        mass = UsdPhysics.MassAPI(prim)
        if not mass.GetMassAttr().HasAuthoredValue() or mass.GetMassAttr().Get() <= 0:
            continue
        body = spec.body(prim.GetName())
        if body is None:
            continue
        if not authored_inertial(mass):
            index = geometric.body(prim.GetName()).id
            if geometric.body_mass[index] <= 0:
                raise ValueError(f'{prim.GetPath()}: mass without inertia needs collision geometry')
            body.mass = mass.GetMassAttr().Get()
            body.ipos, body.iquat = geometric.body_ipos[index], geometric.body_iquat[index]
            body.inertia = geometric.body_inertia[index] * body.mass / geometric.body_mass[index]
            body.fullinertia = [math.nan] * 6
            body.explicitinertial = True
            updated.append(prim.GetName())
            continue
        transform = cache.GetLocalToWorldTransform(prim)
        rotation = transform.RemoveScaleShear().ExtractRotationQuat() * Gf.Quatd(mass.GetPrincipalAxesAttr().Get())
        centre = transform.Transform(Gf.Vec3d(mass.GetCenterOfMassAttr().Get()))
        index = model.body(prim.GetName()).id
        frame = data.xmat[index].reshape(3, 3)
        principal = np.zeros(4)
        mujoco.mju_mat2Quat(principal, np.array(Gf.Matrix3d(rotation).GetTranspose()).reshape(9))
        inverse, local = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, data.xquat[index])
        mujoco.mju_mulQuat(local, inverse, principal)
        body.mass = mass.GetMassAttr().Get()
        body.ipos = frame.T @ (np.array(centre) - data.xpos[index])
        body.iquat = local
        body.inertia = list(mass.GetDiagonalInertiaAttr().Get())
        body.fullinertia = [math.nan] * 6
        body.explicitinertial = True
        updated.append(prim.GetName())
    spec.compile()
    Path(mjcf).write_text(spec.to_xml())
    return updated


# URDF/MJCF -> USD (Isaac Sim importers) ------------------------------------

def to_usd(source, directory, name, fix_base):
    """Run the Isaac Lab importer in the Isaac interpreter (it starts a headless Kit app)."""
    python = os.environ.get('ISAAC_PYTHON', str(ROOT / '.venv/bin/python'))
    command = [python, '-m', 'shared.asset_formats', '_isaac_import', str(source), str(directory), name]
    if fix_base:
        command.append('--fix-base')
    subprocess.run(command, check=True, cwd=ROOT, env={
        **os.environ, 'OMNI_KIT_ACCEPT_EULA': 'YES',
        'PYTHONPATH': os.pathsep.join(filter(None, (str(ROOT), os.environ.get('PYTHONPATH'))))})
    return Path(directory) / (name + '.usd'), json.loads((Path(directory) / '.import.json').read_text())


def _isaac_import(source, directory, name, fix_base):
    from isaaclab.app import AppLauncher
    application = AppLauncher(headless=True).app
    try:
        from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg, UrdfConverter, UrdfConverterCfg
        common = dict(asset_path=str(source), usd_dir=str(directory), usd_file_name=name + '.usd',
                      fix_base=fix_base, make_instanceable=False, force_usd_conversion=True)
        if source_format(source) == 'urdf':
            # Same fixed links as the MJCF; drives are set by the backend factory.
            UrdfConverter(UrdfConverterCfg(**common, merge_fixed_joints=False, joint_drive=None))
            tool = 'Isaac Lab UrdfConverter'
        else:
            MjcfConverter(MjcfConverterCfg(**common))
            tool = 'Isaac Lab MjcfConverter'
        import isaacsim
        (Path(directory) / '.import.json').write_text(json.dumps(
            {'tool': tool, 'version': getattr(isaacsim, '__version__', 'unknown'), 'fix_base': fix_base}))
    finally:
        application.close()


# Alignment -----------------------------------------------------------------

def usd_bodies(path):
    """Rest-pose rigid bodies and joints read directly with pxr (fixed-joint children as PhysX poses them)."""
    from pxr import Gf, UsdGeom, UsdPhysics
    stage = physx_stage(path)[0]
    cache = UsdGeom.XformCache()
    bodies, joints = {}, {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            transform = cache.GetLocalToWorldTransform(prim)
            mass = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
            rotation = transform.RemoveScaleShear().ExtractRotationQuat()
            entry = {'position': list(transform.ExtractTranslation()),
                     'orientation': [rotation.GetReal(), *rotation.GetImaginary()]}
            # UsdPhysics reads mass <= 0 as unset: PhysX then integrates the colliders at their density.
            if mass is not None and mass.GetMassAttr().HasAuthoredValue() and mass.GetMassAttr().Get() > 0:
                entry['mass'] = mass.GetMassAttr().Get()
            if 'mass' in entry and authored_inertial(mass):
                entry['com'] = list(transform.Transform(Gf.Vec3d(mass.GetCenterOfMassAttr().Get())))
                entry['inertia'] = sorted(mass.GetDiagonalInertiaAttr().Get())
            bodies[prim.GetName()] = entry
        for kind, schema in (('hinge', UsdPhysics.RevoluteJoint), ('slide', UsdPhysics.PrismaticJoint)):
            if prim.IsA(schema):
                joint = schema(prim)
                parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
                rotation = (cache.GetLocalToWorldTransform(parent).RemoveScaleShear().ExtractRotationQuat()
                            * Gf.Quatd(joint.GetLocalRot0Attr().Get()))
                axis = {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[joint.GetAxisAttr().Get()]
                scale = math.pi / 180 if kind == 'hinge' else 1.0
                limits = [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]
                continuous = all(value is None or math.isinf(value) or abs(value) >= 3e38 for value in limits)
                joints[prim.GetName()] = {
                    'type': kind, 'axis': list(Gf.Rotation(rotation).TransformDir(Gf.Vec3d(*axis))),
                    'range': None if continuous else [value * scale for value in limits]}
    for prim, body in usd_colliders(stage):
        if body is not None and body.GetName() in bodies:
            colliders = bodies[body.GetName()].setdefault('colliders', {})
            kind = COLLIDER_TYPES.get(prim.GetTypeName(), prim.GetTypeName())
            colliders[kind] = colliders.get(kind, 0) + 1
    for entry in bodies.values():
        if 'mass' not in entry and not entry.get('colliders'):
            entry['mass'], entry['physx_default_mass'] = PHYSX_DEFAULT_MASS, True
    return bodies, joints


def mjcf_bodies(path):
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    bodies = {model.body(index).name: {
        'position': data.xpos[index].tolist(), 'orientation': data.xquat[index].tolist(),
        'mass': float(model.body_mass[index]),
        'com': data.xipos[index].tolist(), 'inertia': sorted(model.body_inertia[index].tolist())}
        for index in range(1, model.nbody)}
    shapes = {int(getattr(mujoco.mjtGeom, 'mjGEOM_' + kind.upper())): kind
              for kind in ('box', 'sphere', 'cylinder', 'capsule', 'mesh')}
    for index in range(model.ngeom):
        if (model.geom_contype[index] or model.geom_conaffinity[index]) and model.geom_bodyid[index]:
            colliders = bodies[model.body(model.geom_bodyid[index]).name].setdefault('colliders', {})
            kind = shapes.get(int(model.geom_type[index]), str(model.geom_type[index]))
            colliders[kind] = colliders.get(kind, 0) + 1
    kinds = {mujoco.mjtJoint.mjJNT_HINGE: 'hinge', mujoco.mjtJoint.mjJNT_SLIDE: 'slide'}
    joints = {model.joint(index).name: {
        'type': kinds.get(model.jnt_type[index], 'other'), 'axis': data.xaxis[index].tolist(),
        'range': model.jnt_range[index].tolist() if model.jnt_limited[index] else None}
        for index in range(model.njnt)}
    return bodies, joints


def usd_joint_order(path):
    """Movable joints in PhysX articulation order: breadth-first from the root link, the links of
    one depth sorted by name. This is Isaac's joint_names order for every asset recorded here."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    edges = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        ends = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if all(ends) and not joint.GetExcludeFromArticulationAttr().Get():
            movable = prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
            edges.append((ends[0][0], ends[1][0], prim.GetName(), movable))
    if not edges:
        return []
    roots = {parent for parent, *_ in edges} - {child for _, child, *_ in edges}
    if len(roots) != 1:
        raise ValueError(f'{path}: expected one articulation root link, found {sorted(map(str, roots))}')
    order, level = [], sorted(roots)
    while level:
        below = sorted((edge for edge in edges if edge[0] in level), key=lambda edge: edge[1].name)
        order += [name for _, _, name, movable in below if movable]
        level = [child for _, child, *_ in below]
    return order


def mjcf_joint_order(path):
    """Hinge and slide joints in MuJoCo's order (depth-first through the body tree)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(path))
    return [model.joint(index).name for index in range(model.njnt)
            if model.jnt_type[index] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]


def joint_order_problems(record):
    """The recorded joint_order must name exactly the movable joints of every format."""
    found = {format_: (usd_joint_order if format_ == 'usd' else mjcf_joint_order)(ROOT / path)
             for format_, path in record['formats'].items() if format_ in ('usd', 'mjcf')}
    order = record.get('joint_order')
    if order is None:
        return ['ASSET.json has no joint_order; rerun run.py asset convert'] if any(found.values()) else []
    problems = [f'joint_order repeats {name}' for name, count in Counter(order).items() if count > 1]
    for format_, names in found.items():
        missing, extra = sorted(set(order) - set(names)), sorted(set(names) - set(order))
        if missing or extra:
            problems.append(f'{format_} joints differ from joint_order: missing {missing}, not listed {extra}')
    return problems


def recorded_joint_order(path):
    """Canonical joint order of the asset whose ASSET.json lists `path` as its source or a format;
    None for files no ASSET.json lists. Backends present joint readings in this order."""
    path = Path(path).resolve()
    for record_path in (directory / 'ASSET.json' for directory in path.parents):
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        if path not in {(ROOT / item).resolve() for item in (record['source'], *record['formats'].values())}:
            continue
        if 'joint_order' not in record:
            raise ValueError(f'{workspace_path(record_path)} has no joint_order; rerun: '
                             f'run.py asset convert {record["source"]} <asset directory>')
        return tuple(record['joint_order'])
    return None


def compare(usd, mjcf):
    """Compare the rest pose of both files; every mismatch is listed, nothing is fixed silently."""
    usd_body, usd_joint = usd_bodies(usd)
    mj_body, mj_joint = mjcf_bodies(mjcf)
    mismatches, worst = [], Counter()

    def check(kind, where, left, right, scale=1.0):
        difference = max(abs(a - b) for a, b in zip(left, right)) / scale
        worst[kind] = max(worst[kind], difference)
        if difference > TOLERANCE[kind]:
            mismatches.append({'item': where, 'kind': kind, 'usd': left, 'mjcf': right})

    for name in sorted(usd_body.keys() & mj_body.keys()):
        a, b = usd_body[name], mj_body[name]
        check('position', f'body {name}', a['position'], b['position'])
        # Mounts (cameras, tool frames) are authored in body frames, so frames must match too.
        dot = min(1., abs(sum(x * y for x, y in zip(a['orientation'], b['orientation']))))
        check('angle', f'body {name} frame orientation', [2 * math.acos(dot)], [0.])
        if 'mass' in a:
            check('mass', f'body {name}', [a['mass']], [b['mass']], max(a['mass'], 1e-6))
            if a['mass'] > 0 and 'com' in a:
                check('position', f'body {name} centre of mass', a['com'], b['com'])
                check('inertia', f'body {name}', a['inertia'], b['inertia'], max(*a['inertia'], 1e-9))
        if a.get('colliders', {}) != b.get('colliders', {}):  # converters can drop shapes (URDF has no capsule)
            mismatches.append({'item': f'body {name} colliders', 'kind': 'colliders',
                               'usd': a.get('colliders', {}), 'mjcf': b.get('colliders', {})})
    for name in sorted(usd_joint.keys() | mj_joint.keys()):
        if name not in usd_joint or name not in mj_joint:
            mismatches.append({'item': f'joint {name}', 'kind': 'missing',
                               'usd': name in usd_joint, 'mjcf': name in mj_joint})
            continue
        a, b = usd_joint[name], mj_joint[name]
        if a['type'] != b['type'] or (a['range'] is None) != (b['range'] is None):
            mismatches.append({'item': f'joint {name}', 'kind': 'type/limit', 'usd': a, 'mjcf': b})
            continue
        check('axis', f'joint {name}', a['axis'], b['axis'])
        if a['range'] is not None:  # both continuous otherwise
            check('angle', f'joint {name} range', a['range'], b['range'])
    for name in sorted(usd_body.keys() - mj_body.keys()):
        if usd_body[name].get('physx_default_mass'):
            mismatches.append({'item': f'body {name} (USD only): no mass and no colliders, PhysX assigns '
                                       f'{PHYSX_DEFAULT_MASS} kg', 'kind': 'mass', 'usd': [PHYSX_DEFAULT_MASS], 'mjcf': None})
    massive = lambda bodies, name: bodies[name].get('mass', 1.0) > 0
    return {
        'aligned': not mismatches,
        'tolerance': TOLERANCE,
        'worst': dict(worst),
        'bodies_compared': len(usd_body.keys() & mj_body.keys()),
        'joints': sorted(usd_joint.keys() & mj_joint.keys()),
        'usd_only_bodies': sorted(name for name in usd_body.keys() - mj_body.keys() if massive(usd_body, name)),
        'mjcf_only_bodies': sorted(name for name in mj_body.keys() - usd_body.keys() if massive(mj_body, name)),
        'mismatches': mismatches,
        'masses_from_colliders': sorted(name for name, body in usd_body.items() if 'mass' not in body),
        'not_compared': ['masses_from_colliders (each engine integrates its own collision shapes at a density)', 'collision shape sizes (MuJoCo uses mesh convex hulls; PhysX follows the USD collider '
                         'approximation); collider counts per body and type are compared', 'contact/friction parameters', 'actuators (declared by backend factories)'],
    }


# Command line --------------------------------------------------------------

def usd_layers(source):
    """Hashes of every layer the USD source composes (references usually point elsewhere)."""
    from pxr import UsdUtils
    layers, _, _ = UsdUtils.ComputeAllDependencies(str(source))
    return {workspace_path(layer.realPath): digest(layer.realPath) for layer in layers if layer.realPath}


def convert(source, output, name=None, packages=None, root=None, fix_base=False, keep_visuals=False,
            forms=('usd', 'mjcf')):
    """Write the missing forms among `forms`; alignment needs both USD and MJCF."""
    source, output = Path(source).resolve(), Path(output).resolve()
    name = name or source.stem
    kind = source_format(source)
    targets = {format_: output / format_ for format_ in ('usd', 'mjcf') if format_ in forms or format_ == kind}
    for format_, directory in targets.items():
        if format_ != kind and directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'{directory} is not empty; remove it or choose another output')
    if (output / 'ASSET.json').exists():
        raise FileExistsError(f'{output / "ASSET.json"} exists; remove it or choose another output')
    steps, formats = [], {kind: workspace_path(source)}
    with tempfile.TemporaryDirectory() as scratch:
        urdf = source if kind == 'urdf' else None
        if kind == 'usd':
            urdf, step = usd_to_urdf(source, scratch, name, root)
            steps.append({'from': 'usd', 'to': 'urdf', **step})
        if kind != 'mjcf' and 'mjcf' in targets:
            path, step = urdf_to_mjcf(urdf, targets['mjcf'], name, packages if kind == 'urdf' else None,
                                      keep_visuals)
            if kind == 'usd':
                step['body_frames_from_usd'] = usd_frames_into_mjcf(path, source)
                step['capsule_colliders_from_usd'] = usd_capsules_into_mjcf(path, source)
                step['inertial_from_usd_mass_api'] = usd_inertia_into_mjcf(path, source)
            steps.append({'from': 'urdf', 'to': 'mjcf', **step})
            formats['mjcf'] = workspace_path(path)
        if kind != 'usd' and 'usd' in targets:
            path, step = to_usd(source, targets['usd'], name, fix_base)
            steps.append({'from': kind, 'to': 'usd', **step})
            formats['usd'] = workspace_path(path)
    record = {
        'source': workspace_path(source), 'source_format': kind, 'source_sha256': digest(source),
        **({'source_layers_sha256': usd_layers(source)} if kind == 'usd' else {}),
        'formats': formats, 'steps': steps,
        # Joint readings use this order in every backend (Isaac's own order for a USD).
        'joint_order': (usd_joint_order(ROOT / formats['usd']) if 'usd' in formats
                        else mjcf_joint_order(ROOT / formats['mjcf'])),
        'alignment': (compare(ROOT / formats['usd'], ROOT / formats['mjcf'])
                      if formats.keys() == {'usd', 'mjcf'} else None),
        'sha256': {path.relative_to(output).as_posix(): digest(path)
                   for format_ in targets if format_ != kind
                   for path in sorted(targets[format_].rglob('*')) if path.is_file()},
    }
    (output / 'ASSET.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def source_hashes(source, packages=None):
    """Content hashes that determine a conversion: USD layers, or a URDF and its meshes."""
    source = Path(source).resolve()
    if source_format(source) == 'usd':
        return usd_layers(source)
    hashes = {workspace_path(source): digest(source)}
    if source_format(source) == 'urdf':
        packages = {key: Path(value).resolve() for key, value in (packages or {}).items()}
        for mesh in ElementTree.parse(source).getroot().iter('mesh'):
            path = resolve_mesh(mesh.get('filename'), source, packages)
            for item in (path, path.with_suffix('.mtl')):
                if item.is_file():
                    hashes[workspace_path(item)] = digest(item)
    return hashes


def upstream_files(source, packages):
    """Files a USD, URDF or MJCF model loads (plus each named ROS package.xml), and references left unresolved."""
    if source_format(source) == 'usd':
        from pxr import UsdUtils
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(source))
        return {Path(layer.realPath) for layer in layers if layer.realPath} | {Path(item) for item in assets}, unresolved
    if source_format(source) == 'mjcf':
        return mjcf_files(source)
    files, missing = {source}, []
    for mesh in ElementTree.parse(source).getroot().iter('mesh'):
        path = resolve_mesh(mesh.get('filename'), source, packages)
        if not path.is_file():
            missing.append(mesh.get('filename'))
        files |= {item for item in (path, path.with_suffix('.mtl')) if item.is_file()}
    return files | {directory / 'package.xml' for directory in packages.values()}, missing


def mjcf_files(source):
    """Files an MJCF model loads: its <include>s and the mesh/texture/hfield/skin files they name.

    Paths follow MuJoCo: includes are relative to the main file, assets to the compiler's
    meshdir/texturedir (assetdir sets both), each relative to the main file."""
    source = Path(source).resolve()
    documents, pending = [], [source]
    while pending:
        path = pending.pop()
        documents.append(path)
        pending += [source.parent / item.get('file') for item in ElementTree.parse(path).getroot().iter('include')]
    directories = {}
    for path in documents:
        for compiler in ElementTree.parse(path).getroot().iter('compiler'):
            for key in ('assetdir', 'meshdir', 'texturedir'):
                if compiler.get(key) is not None:
                    directories[key] = compiler.get(key)
    meshes = directories.get('meshdir', directories.get('assetdir', ''))
    textures = directories.get('texturedir', directories.get('assetdir', ''))
    files, missing = set(documents), []
    for path in documents:
        for element in ElementTree.parse(path).getroot().iter():
            names = [element.get('file')] if element.tag in ('mesh', 'hfield', 'skin', 'texture') else []
            if element.tag == 'texture':
                names += [element.get(key) for key in ('fileright', 'fileleft', 'fileup', 'filedown',
                                                       'filefront', 'fileback')]
            for name in filter(None, names):
                target = source.parent / (textures if element.tag == 'texture' else meshes) / name
                (files.add if target.is_file() else missing.append)(target.resolve() if target.is_file() else name)
    return files, missing


def _git(directory, *args):
    return subprocess.run(['git', '-C', str(directory), *args], check=True, capture_output=True, text=True).stdout.strip()


def vendor(source, output, packages=None):
    """Copy an upstream USD/URDF/MJCF model and every file it loads to OUTPUT/source/<repository path>.

    Files keep their repository paths, so relative USD references and package:// URIs (via the
    copied package.xml) still resolve. OUTPUT/source/SOURCE.json records the repository, commit
    and sha256 of each file; `asset check` verifies them. Only committed, unmodified files are copied.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    packages = {key: Path(value).resolve() for key, value in (packages or {}).items()}
    files, unresolved = upstream_files(source, packages)
    top = Path(_git(source.parent, 'rev-parse', '--show-toplevel'))
    outside = sorted(str(path) for path in files if not path.resolve().is_relative_to(top))
    if outside:
        raise SystemExit(f'{source} loads files outside its repository {top}: {outside}')
    folders = [source.parent, *(folder for folder in source.parent.parents if folder.is_relative_to(top))]
    files |= {path for folder in folders for path in folder.iterdir()
              if path.is_file() and path.name.upper().startswith(('LICENSE', 'COPYING'))}
    relative = sorted(path.resolve().relative_to(top).as_posix() for path in files)
    changed = _git(top, 'status', '--porcelain', '--', *relative)
    if changed:
        raise SystemExit(f'Uncommitted upstream changes; commit or revert them first:\n{changed}')
    url = _git(top, 'remote', 'get-url', 'origin')
    if 'https://' in url:  # drop mirror prefixes such as https://proxy/https://github.com/...
        url = 'https://' + url.rsplit('https://', 1)[1]
    record_path = output / 'source' / 'SOURCE.json'
    record = {'repository': url.removesuffix('.git'), 'commit': _git(top, 'rev-parse', 'HEAD'),
              'note': 'Unmodified upstream files at their repository paths; '
                      "the URDF's package:// URIs resolve against this directory.", 'sha256': {}}
    if record_path.is_file():
        existing = json.loads(record_path.read_text())
        if (existing.get('repository'), existing.get('commit')) != (record['repository'], record['commit']):
            raise SystemExit(f"{record_path} records {existing.get('repository') or existing.get('url')}"
                             f"@{existing.get('commit')}; "
                             'one asset vendors one upstream commit')
        record = existing
    for name in relative:
        hash_ = digest(top / name)
        if record['sha256'].get(name, hash_) != hash_:
            raise SystemExit(f'{record_path.parent / name} already holds different content')
        target = record_path.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(top / name, target)
        record['sha256'][name] = hash_
    record['sha256'] = dict(sorted(record['sha256'].items()))
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return record, workspace_path(record_path.parent / source.relative_to(top)), unresolved


NUCLEUS_URL = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1'


def nucleus_copy(asset, path):
    """Vendored copy of NUCLEUS_URL/PATH in assets/ASSET/source/ (the Isaac release our Isaac Lab loads)."""
    file = ROOT / 'assets' / asset / 'source' / path
    if not file.is_file():
        raise FileNotFoundError(f'{workspace_path(file)} is missing; download it with: '
                                f'run.py asset vendor {NUCLEUS_URL}/{path} assets/{asset} --base {NUCLEUS_URL}')
    return file


def _download(url, attempts=3):
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=120) as response:
                return response.read()
        except HTTPError:
            raise
        except (URLError, OSError):
            if attempt == attempts - 1:
                raise


def _exists(url):
    """False when the server reports no such file (S3 answers 403 for missing keys); other errors raise."""
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    try:
        with urlopen(Request(url, method='HEAD'), timeout=60):
            return True
    except HTTPError as error:
        if error.code in (403, 404):
            return False
        raise
    except URLError as error:
        if isinstance(error.reason, FileNotFoundError):
            return False
        raise


def url_dependencies(path):
    """Relative references a downloaded file makes: USD layers/assets, or the textures an MDL material names."""
    if path.suffix in USD_SUFFIXES:
        from pxr import UsdUtils
        return [item for group in UsdUtils.ExtractExternalReferences(str(path)) for item in group]
    if path.suffix == '.mdl':
        return re.findall(r'texture_2d\(\s*"([^"]+)"', path.read_text(errors='ignore'))
    return []


def vendor_url(url, output, base=None):
    """Download a model (and every file it references, recursively) to OUTPUT/source/<path below BASE>.

    BASE is the source root recorded in SOURCE.json, by default the URL's scheme and host; for
    Isaac Nucleus pass the release directory (…/Assets/Isaac/5.1). References that are absolute,
    point outside BASE, or do not exist there (Kit's built-in MDLs such as OmniPBR.mdl) are not
    copied and are returned as unresolved."""
    from urllib.parse import urljoin, urlsplit
    output = Path(output).resolve()
    parts = urlsplit(url)
    base = (base or f'{parts.scheme}://{parts.netloc}').rstrip('/') + '/'
    if not url.startswith(base):
        raise SystemExit(f'{url} is not below --base {base}')
    record_path = output / 'source' / 'SOURCE.json'
    record = {'url': base.rstrip('/'), 'note': 'Unmodified downloads at their paths below url; '
                                               'relative references resolve within this directory.', 'sha256': {}}
    if record_path.is_file():
        existing = json.loads(record_path.read_text())
        if existing.get('url') != record['url']:
            raise SystemExit(f"{record_path} records {existing.get('repository') or existing.get('url')}; "
                             'one asset vendors one source')
        record = existing
    pending, seen, unresolved = [url], set(), []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        name = current[len(base):]
        content = _download(current)
        hash_ = hashlib.sha256(content).hexdigest()
        if record['sha256'].get(name, hash_) != hash_:
            raise SystemExit(f'{record_path.parent / name} already holds different content')
        target = record_path.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        record['sha256'][name] = hash_
        print(f'{name}: {len(content)} bytes', flush=True)
        for reference in url_dependencies(target):
            reference = reference.replace('<UDIM>', '1001')
            dependency = urljoin(current, reference)
            if '://' in reference or not dependency.startswith(base):
                unresolved.append(reference)
                continue
            if dependency in seen:
                continue
            if not _exists(dependency):
                unresolved.append(reference)
                continue
            pending.append(dependency)
    record['sha256'] = dict(sorted(record['sha256'].items()))
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return record, workspace_path(record_path.parent / url[len(base):]), sorted(set(unresolved))


def source_problems(directory):
    """Files listed in DIRECTORY/source/SOURCE.json that are missing or changed (None without the record)."""
    record_path = Path(directory) / 'source' / 'SOURCE.json'
    if not record_path.is_file():
        return None
    expected = json.loads(record_path.read_text())['sha256']
    return [f'{"changed" if (record_path.parent / name).is_file() else "missing"} source/{name}'
            for name, hash_ in expected.items()
            if not (record_path.parent / name).is_file() or digest(record_path.parent / name) != hash_]


def recorded_mjcf(source):
    """MJCF recorded by `asset convert` next to a USD/URDF source, or None.

    A record whose source changed afterwards is an error, never silently reused."""
    source = Path(source).resolve()
    for record_path in (directory / 'ASSET.json' for directory in source.parents):
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        if ROOT / record['source'] != source or 'mjcf' not in record['formats']:
            continue
        if record['source_format'] == 'usd':
            current, recorded = usd_layers(source), record['source_layers_sha256']
        else:
            current, recorded = digest(source), record['source_sha256']
        if current != recorded:
            raise ValueError(f'{source} changed after {workspace_path(record_path)} was written; '
                             f'rerun: run.py asset convert {record["source"]} <asset directory>')
        # Compare again rather than trust the stored flag: checks may have grown since.
        if 'usd' in record['formats'] and not compare(ROOT / record['formats']['usd'],
                                                      ROOT / record['formats']['mjcf'])['aligned']:
            raise ValueError(f'{workspace_path(record_path)}: recorded MJCF is not aligned with the USD; '
                             f'check: run.py asset check {workspace_path(record_path.parent)}')
        return ROOT / record['formats']['mjcf']
    return None


def usd_drives(path):
    """Joint drives authored in a USD articulation, per radian or metre (UsdPhysics angular units are degrees)."""
    from pxr import Usd, UsdPhysics
    drives = {}
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        for kind, joint_type, scale in (('angular', UsdPhysics.RevoluteJoint, 180 / math.pi),
                                        ('linear', UsdPhysics.PrismaticJoint, 1.)):
            if not (prim.IsA(joint_type) and prim.HasAPI(UsdPhysics.DriveAPI, kind)):
                continue
            drive = UsdPhysics.DriveAPI(prim, kind)
            if drive.GetTypeAttr().Get() not in (None, 'force'):
                raise NotImplementedError(f'{prim.GetPath()}: only force drives map to MuJoCo position actuators')
            limit = drive.GetMaxForceAttr().Get()
            drives[prim.GetName()] = {
                'stiffness': (drive.GetStiffnessAttr().Get() or 0.) * scale,
                'damping': (drive.GetDampingAttr().Get() or 0.) * scale,
                'max_force': None if limit is None or limit >= 3e38 or math.isinf(limit) else limit}
    return drives


def mjcf_for(source, *, packages=None, root=None):
    """MJCF form of a USD/URDF asset for MuJoCo backends.

    Uses the conversion recorded in ASSET.json beside the source; otherwise converts once
    into .cache/asset_mjcf keyed by source content, conversion options and this module.
    A USD conversion that does not align with its source raises instead of loading."""
    source = Path(source).resolve()
    kind = source_format(source)
    if kind == 'mjcf':
        return source
    if root is None and not packages:
        recorded = recorded_mjcf(source)
        if recorded is not None:
            return recorded
    key = hashlib.sha256(json.dumps({
        'source': source_hashes(source, packages), 'root': root,
        'packages': {key: str(Path(value).resolve()) for key, value in (packages or {}).items()},
        'converter': digest(__file__)}, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / '.cache/asset_mjcf' / f'{source.stem}_{key}'
    if not (output / 'ASSET.json').is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=output.name + '.', dir=output.parent))
        try:
            record = convert(source, staging, packages=packages, root=root, forms=('mjcf',))
            if record['alignment'] is not None and not record['alignment']['aligned']:
                raise ValueError(f'MJCF converted from {workspace_path(source)} does not align:\n'
                                 + summary(record['alignment']))
            record['formats']['mjcf'] = workspace_path(output / 'mjcf' / f'{source.stem}.xml')
            (staging / 'ASSET.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
            try:
                staging.rename(output)
            except OSError:
                if not (output / 'ASSET.json').is_file():
                    raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return output / 'mjcf' / f'{source.stem}.xml'


def summary(report):
    lines = [f"aligned: {report['aligned']}  bodies {report['bodies_compared']}  joints {len(report['joints'])}",
             '  worst: ' + ', '.join(f'{key} {value:.2e}' for key, value in report['worst'].items())]
    for key in ('usd_only_bodies', 'mjcf_only_bodies', 'masses_from_colliders'):
        if report.get(key):
            lines.append(f'  {key}: {", ".join(report[key])}')
    lines += [f"  mismatch {item['kind']}: {item['item']} usd={item['usd']} mjcf={item['mjcf']}"
              for item in report['mismatches']]
    return '\n'.join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['_isaac_import']:
        source, directory, name = argv[1:4]
        return _isaac_import(Path(source), Path(directory), name, '--fix-base' in argv)
    if importlib.util.find_spec('mujoco') is None:
        interpreter = ROOT / '.venv-mjlab/bin/python'
        if not interpreter.is_file():
            raise SystemExit('run.py asset needs an interpreter with mujoco and pxr (e.g. .venv-mjlab)')
        os.execv(str(interpreter), [str(interpreter), str(ROOT / 'run.py'), 'asset', *argv])
    parser = argparse.ArgumentParser(prog='run.py asset', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('convert', help='Create the missing USD/MJCF form of an asset')
    create.add_argument('source', type=Path, help='USD, URDF or MJCF file')
    create.add_argument('output', type=Path, help='Asset directory; writes usd/ and/or mjcf/ and ASSET.json')
    create.add_argument('--name', help='File name of the generated model (default: source stem)')
    create.add_argument('--package', action='append', default=[], metavar='NAME=PATH',
                        help='URDF package:// root (default: nearest package.xml with that name)')
    create.add_argument('--root', help='USD source: prim path of the kinematic root')
    create.add_argument('--fix-base', action='store_true', help='URDF/MJCF source: fixed-base articulation in USD')
    create.add_argument('--keep-visuals', action='store_true',
                        help='Keep visual meshes in MJCF for every link; collision geoms then go to the hidden group')
    copy = commands.add_parser('vendor', help='Copy an upstream USD/URDF/MJCF and the files it loads into ASSET/source/')
    copy.add_argument('source', help='USD, URDF or MJCF file inside a git checkout, or an http(s) URL')
    copy.add_argument('output', type=Path, help='Asset directory; writes source/<upstream path> and SOURCE.json')
    copy.add_argument('--base', help='URL source: root recorded in SOURCE.json; files keep their path below it')
    copy.add_argument('--package', action='append', default=[], metavar='NAME=PATH',
                      help='URDF package:// root (default: nearest package.xml with that name)')
    verify = commands.add_parser('check', help='Verify vendored sources and compare the USD and MJCF forms')
    verify.add_argument('asset', type=Path, help='Asset directory with ASSET.json and/or source/SOURCE.json, '
                                                 'or a USD file')
    verify.add_argument('mjcf', type=Path, nargs='?', help='MJCF file when the first argument is a USD file')
    tile = commands.add_parser('terrain', help='Generate a terrain tile as an ordinary USD + MJCF asset')
    tile.add_argument('variant', help='Terrain variant name (shared/backends/isaac/assets.py TERRAIN_VARIANTS), '
                                      'or "all" for every preset into OUTPUT/<variant>')
    tile.add_argument('output', type=Path, help='Asset directory; writes usd/, mjcf/ and ASSET.json')
    tile.add_argument('--name', help='File name of the generated model (default: variant)')
    tile.add_argument('--size', type=float, default=8.0, help='Square tile edge in metres (default 8)')
    tile.add_argument('--difficulty', type=float, default=0.5, help='Isaac Lab difficulty in [0, 1] (default 0.5)')
    tile.add_argument('--seed', type=int, default=0)
    tile.add_argument('--param', metavar='JSON',
                      help='Sub-terrain parameters as a JSON object (default: shared/terrain_assets.py PRESETS)')
    options = parser.parse_args(argv)
    if options.command == 'terrain':
        from shared.terrain_assets import PRESETS, generate
        parameters = None if options.param is None else json.loads(options.param)
        if options.variant == 'all':
            if options.name or parameters is not None:
                parser.error('"all" uses each preset; --name and --param apply to a single variant')
            jobs = [(variant, options.output / variant) for variant in PRESETS
                    if not (options.output / variant / 'ASSET.json').exists()]
        else:
            jobs = [(options.variant, options.output)]
        for variant, output in jobs:
            record = generate(variant, output, name=options.name, size=options.size, difficulty=options.difficulty,
                              seed=options.seed, parameters=parameters)
            print(json.dumps({**record['formats'], **record['terrain']}, indent=2))
            print(json.dumps(record['steps'][0]['mjcf_collision']),
                  'max height difference', f"{record['steps'][0]['max_height_difference']:.2e} m", flush=True)
        return 0
    packages = {}
    for item in getattr(options, 'package', []):
        key, separator, value = item.partition('=')
        if not separator:
            parser.error('--package expects NAME=PATH')
        packages[key] = value
    if options.command == 'vendor':
        if '://' in options.source:
            record, copied, unresolved = vendor_url(options.source, options.output, options.base)
            origin = record['url']
        else:
            record, copied, unresolved = vendor(options.source, options.output, packages)
            origin = f"{record['repository']}@{record['commit']}"
        print(f"{origin}: {len(record['sha256'])} files; model at {copied}")
        if unresolved:
            print('unresolved (not copied):', ', '.join(map(str, unresolved)))
        return 0
    if options.command == 'convert':
        record = convert(options.source, options.output, options.name, packages, options.root,
                         options.fix_base, options.keep_visuals)
        print(json.dumps(record['formats'], indent=2))
        print(summary(record['alignment']))
        return 0 if record['alignment']['aligned'] else 1
    if options.mjcf is None:
        problems = source_problems(options.asset)
        if problems:
            print('vendored sources differ from source/SOURCE.json:', *problems, sep='\n  ')
            return 1
        if not (options.asset / 'ASSET.json').is_file():
            if problems is None:
                parser.error(f'{options.asset} has neither ASSET.json nor source/SOURCE.json')
            print('vendored sources match source/SOURCE.json; no ASSET.json, so no USD/MJCF comparison')
            return 0
        if problems is not None:
            print('vendored sources match source/SOURCE.json')
        record = json.loads((options.asset / 'ASSET.json').read_text())
        order = joint_order_problems(record)
        if order:
            print('joint order problems:', *order, sep='\n  ')
        else:
            print('joint order:', ', '.join(record.get('joint_order', [])) or 'no movable joints')
        usd, mjcf = ROOT / record['formats']['usd'], ROOT / record['formats']['mjcf']
    else:
        usd, mjcf, order = options.asset, options.mjcf, []
    report = compare(usd, mjcf)
    print(summary(report))
    return 0 if report['aligned'] and not order else 1


if __name__ == '__main__':
    raise SystemExit(main())
