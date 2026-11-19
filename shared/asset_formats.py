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

SimReady assets (assets/simready/<publisher>/<asset>/simready_usd/) are used as USD only, through a
wrapper layer that references them unchanged and authors what Isaac Lab needs (see simready_wrap):

    run.py asset simready assets/simready/nvidia/cabinet_file_a01/simready_usd/sm_cabinet_file_a01_01.usd \
        assets/cabinet_file_a01
"""
import argparse
import contextlib
import functools
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
import time
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


def _download(url, attempts=5):
    from http.client import HTTPException
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=120) as response:
                return response.read()
        except HTTPError:
            raise
        except (URLError, OSError, HTTPException):
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def _exists(url, attempts=5):
    """False when the server reports no such file (S3 answers 403 for missing keys); other errors raise."""
    from http.client import HTTPException
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen
    for attempt in range(attempts):
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
            if attempt == attempts - 1:
                raise
        except (OSError, HTTPException):
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)


def url_dependencies(path):
    """Relative references a downloaded file makes: USD layers/assets, or the textures and relative
    modules (`import .::Core::*`, `using ..::lib::Base import *`) an MDL material names."""
    if path.suffix in USD_SUFFIXES:
        from pxr import UsdUtils
        return [item for group in UsdUtils.ExtractExternalReferences(str(path)) for item in group]
    if path.suffix == '.mdl':
        text = path.read_text(errors='ignore')
        modules = []
        for kind, module, star in re.findall(r'\b(import|using)\s+(\.{1,2}(?:::\w+)+)(::\*)?', text):
            parts = module.split('::')
            if kind == 'import' and not star:
                parts = parts[:-1]  # import .::module::symbol
            modules.append('/'.join(parts) + '.mdl')
        return re.findall(r'texture_2d\(\s*"([^"]+)"', text) + modules
    return []


def vendor_url(url, output, base=None, threads=16):
    """Download a model (and every file it references, recursively) to OUTPUT/source/<path below BASE>.

    BASE is the source root recorded in SOURCE.json, by default the URL's scheme and host; for
    Isaac Nucleus pass the release directory (…/Assets/Isaac/5.1). References that are absolute,
    point outside BASE, or do not exist there (Kit's built-in MDLs such as OmniPBR.mdl) are not
    copied and are returned as unresolved. Files are fetched THREADS at a time, one reference
    level after another; SOURCE.json is saved after every level, and files it already records
    with matching content are not downloaded again, so an interrupted run resumes."""
    from concurrent.futures import ThreadPoolExecutor
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

    def fetch(current):
        name = current[len(base):]
        target = record_path.parent / name
        if name in record['sha256'] and target.is_file() and digest(target) == record['sha256'][name]:
            return target, record['sha256'][name]
        content = _download(current)
        hash_ = hashlib.sha256(content).hexdigest()
        if record['sha256'].get(name, hash_) != hash_:
            raise SystemExit(f'{target} already holds different content')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f'{name}: {len(content)} bytes', flush=True)
        return target, hash_

    def save():
        record['sha256'] = dict(sorted(record['sha256'].items()))
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + '\n')

    level, seen, unresolved = [url], {url}, []
    with ThreadPoolExecutor(threads) as pool:
        while level:
            candidates = {}
            for current, (target, hash_) in zip(level, pool.map(fetch, level)):
                record['sha256'][current[len(base):]] = hash_
                for reference in url_dependencies(target):
                    reference = reference.replace('<UDIM>', '1001')
                    dependency = urljoin(current, reference)
                    if '://' in reference or not dependency.startswith(base):
                        unresolved.append(reference)
                    elif dependency not in seen:
                        candidates.setdefault(dependency, reference)
            save()
            seen |= set(candidates)
            level = []
            for (dependency, reference), present in zip(candidates.items(), pool.map(_exists, candidates)):
                (level if present else unresolved).append(dependency if present else reference)
    save()
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

COLLIDER_SUFFIX = '_simplified_collider'  # beside a gprim scene object (_simplify_colliders)


def pruned_layer(path, exclude):
    """``scene_edit_layer`` removing ``exclude`` only."""
    return scene_edit_layer(path, exclude)


def _pose_matrix(pose):
    from pxr import Gf
    w, x, y, z = (float(v) for v in pose.get('orientation', (1, 0, 0, 0)))
    return Gf.Matrix4d().SetRotate(Gf.Quatd(w, x, y, z).GetNormalized()).SetTranslateOnly(
        Gf.Vec3d(*(float(v) for v in pose['position'])))


def scene_edit_layer(path, exclude=(), moved=None):
    """A cached USD layer over the scene ``path`` with its objects (paths relative to the default prim, see
    shared/assembly/composition.py) edited as ``usd_object`` says: ``exclude`` deactivates them (both engines
    skip inactive prims), ``moved`` (object -> ``{'position', 'orientation'}``, its pose in the frame the
    entity places, as ``object_pose`` reads it) puts them elsewhere, still static. The collider the
    environment wrapper wrote beside a gprim object follows it either way. Returns the layer path."""
    from pxr import Gf, Sdf, Usd, UsdGeom
    path = Path(path).resolve()
    names = sorted({str(item).strip('/') for item in exclude or ()})
    moves = {str(name).strip('/'): pose for name, pose in (moved or {}).items()}
    if any(not name or '..' in name.split('/') for name in [*names, *moves]):
        raise ValueError(f'{workspace_path(path)}: exclude and moved name prims relative to the default prim')
    for name, pose in moves.items():
        if not isinstance(pose, dict) or len(pose.get('position') or ()) != 3 or len(pose.get('orientation', (1, 0, 0, 0))) != 4:
            raise ValueError(f'{workspace_path(path)}: moved.{name} is {{position: [x, y, z], orientation: [w, x, y, z]}}')
    stage = Usd.Stage.Open(str(path))
    top = stage.GetDefaultPrim()
    if not top:
        raise ValueError(f'{workspace_path(path)}: no default prim to edit objects of')
    missing = [name for name in [*names, *moves] if not stage.GetPrimAtPath(top.GetPath().AppendPath(name))]
    if missing:
        raise ValueError(f'{workspace_path(path)}: exclude or moved names prims that do not exist: {", ".join(missing[:5])}')
    key = hashlib.sha256(json.dumps({'source': source_hashes(path), 'exclude': names, 'moved': moves,
                                     'writer': digest(__file__)}, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / '.cache/usd_pruned' / f'{path.stem}_{key}.usda'
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateAnonymous('.usda')
    out = Usd.Stage.Open(layer)
    out.GetRootLayer().subLayerPaths.append(str(path))
    out.SetDefaultPrim(out.OverridePrim(top.GetPath()))

    def with_collider(name):
        prim = stage.GetPrimAtPath(top.GetPath().AppendPath(name))
        beside = stage.GetPrimAtPath(top.GetPath().AppendPath(name + COLLIDER_SUFFIX))  # a gprim object's collider
        return [prim, beside] if beside else [prim]
    for name in names:
        for prim in with_collider(name):
            _editable(out, prim, top).SetActive(False)
    if moves:
        cache = UsdGeom.XformCache()
        into_top = cache.GetLocalToWorldTransform(top).GetInverse() * Gf.Matrix4d().SetScale(Gf.Vec3d(*_composed_top_scale(top)))
        for name, pose in sorted(moves.items()):
            _, current, _ = _object_placement(stage, name)
            motion = current.GetInverse() * _pose_matrix(pose)  # row vectors: old placement -> new
            for prim in with_collider(name):
                parent = cache.GetLocalToWorldTransform(prim.GetParent()) * into_top
                local = UsdGeom.Xformable(prim).GetLocalTransformation() * parent * motion * parent.GetInverse()
                editable = UsdGeom.Xformable(_editable(out, prim, top))
                editable.ClearXformOpOrder()
                editable.AddTransformOp(opSuffix='moved').Set(local)
    del out, stage
    layer.subLayerPaths[0] = os.path.relpath(path, output.parent)
    staging = output.with_suffix(f'.{os.getpid()}.usda')
    layer.Export(str(staging))
    staging.replace(output)
    return output
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateAnonymous('.usda')
    out = Usd.Stage.Open(layer)
    out.GetRootLayer().subLayerPaths.append(str(path))
    out.SetDefaultPrim(out.OverridePrim(top.GetPath()))
    for name in names:
        prim = stage.GetPrimAtPath(top.GetPath().AppendPath(name))
        _editable(out, prim, top).SetActive(False)
        beside = stage.GetPrimAtPath(top.GetPath().AppendPath(name + COLLIDER_SUFFIX))  # a gprim object's collider
        if beside:
            _editable(out, beside, top).SetActive(False)
    del out, stage, prim
    layer.subLayerPaths[0] = os.path.relpath(path, output.parent)
    staging = output.with_suffix(f'.{os.getpid()}.usda')
    layer.Export(str(staging))
    staging.replace(output)
    return output


# Shader inputs holding a surface's base colour (lower case) and words naming its texture: UsdPreviewSurface,
# OmniPBR, the GRScenes MDL materials (BaseColor_Color x BaseColor_Tex, the texture used when IsBaseColorTex
# is set) and Isaac's MDL instances (Color x BaseColor).
BASE_COLOR_INPUTS = ('diffusecolor', 'diffuse_color_constant', 'basecolor_color', 'base_color', 'basecolor',
                     'albedo', 'color')
BASE_TEXTURE_WORDS = ('basecolor', 'base_color', 'diffuse', 'albedo')


def _texture_average(path, cache):
    if path not in cache:
        cache[path] = None
        with contextlib.suppress(Exception):  # unreadable or missing: the constant colour alone
            from PIL import Image
            with Image.open(path) as image:
                image.draft('RGB', (64, 64))  # JPEG: decode at a fraction of the size
                pixels = image.convert('RGB').resize((8, 8))
                cache[path] = [sum(channel) / (64 * 255.) for channel in zip(*pixels.getdata())]
    return cache[path]


def base_material(prim, cache=None):
    """Base colour of a gprim for the editor's lightweight meshes (not for rendering), or None:
    ``{'color', 'tint', 'texture'}`` with ``color`` (r, g, b in 0-1) the bound material's base colour
    times the average of its base colour texture (else the displayColor), ``tint`` that base colour alone
    and ``texture`` the texture's resolved path (None without one). ``cache`` (a dict) keeps texture
    averages between calls."""
    from pxr import Sdf, UsdGeom, UsdShade
    cache = {} if cache is None else cache
    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
    shader = None
    if material:
        shader = material.ComputeSurfaceSource('mdl')[0] or material.ComputeSurfaceSource()[0]
    if not shader:
        shown = UsdGeom.PrimvarsAPI(prim).GetPrimvar('displayColor')
        values = shown.ComputeFlattened() if shown and shown.HasValue() else None
        if not values:
            return None
        color = [sum(value[i] for value in values) / len(values) for i in range(3)]
        return {'color': color, 'tint': color, 'texture': None}
    inputs = {item.GetBaseName().lower(): item for item in shader.GetInputs()}
    color, texture = None, None
    for name in BASE_COLOR_INPUTS:
        source = inputs.get(name)
        if source is None:
            continue
        if source.HasConnectedSource():  # UsdPreviewSurface: diffuseColor from a UsdUVTexture
            upstream = UsdShade.Shader(source.GetConnectedSources()[0][0].source.GetPrim())
            file = upstream.GetInput('file') if upstream else None
            texture = file.Get() if file else None
            break
        value = source.Get()
        if hasattr(value, '__len__') and len(value) >= 3 and not isinstance(value, (str, Sdf.AssetPath)):
            color = [float(value[i]) for i in range(3)]
            break
    if texture is None and (inputs.get('isbasecolortex') is None or inputs['isbasecolortex'].Get()):
        texture = next((item.Get() for name, item in inputs.items() if any(word in name for word in BASE_TEXTURE_WORDS)
                        and isinstance(item.Get(), Sdf.AssetPath)), None)
    path = texture.resolvedPath if isinstance(texture, Sdf.AssetPath) and texture.resolvedPath else None
    average = _texture_average(path, cache) if path else None
    if color is None and average is None:
        return None
    return {'color': [a * b for a, b in zip(color or (1., 1., 1.), average or (1., 1., 1.))],
            'tint': color or [1., 1., 1.], 'texture': path if average is not None else None}


def base_color(prim, cache=None):
    """``base_material``'s colour alone: the base colour times the average of its texture, or None."""
    material = base_material(prim, cache)
    return material['color'] if material else None


def authored_top_scale(path):
    """Scale the file authors on its default prim (e.g. 0.01, the unit conversion of a centimetre
    environment wrapper), as (x, y, z): Isaac Lab replaces the top prim's transform with the spawn pose
    and scale, so usd_object multiplies this one back in."""
    from pxr import Sdf
    scale = (1., 1., 1.)
    layer = Sdf.Layer.FindOrOpen(str(path))
    top = layer.GetPrimAtPath(layer.defaultPrim) if layer and layer.defaultPrim else None
    order = top.attributes.get('xformOpOrder') if top else None
    for name in (order.default or []) if order else []:
        if name.startswith('xformOp:scale') and (op := top.attributes.get(name)) and op.default is not None:
            authored = op.default
            authored = tuple(authored) if hasattr(authored, '__len__') else (authored,) * 3
            scale = tuple(a * float(b) for a, b in zip(scale, authored))
    return scale


def _composed_top_scale(top):
    """``authored_top_scale`` of an open stage's default prim, through its sublayers and references."""
    from pxr import UsdGeom
    scale = (1., 1., 1.)
    for op in UsdGeom.Xformable(top).GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale and op.Get() is not None:
            scale = tuple(a * float(b) for a, b in zip(scale, op.Get()))
    return scale


def _object_placement(stage, prim):
    """(object prim, its pose in the frame usd_object spawns the file in as a rigid Gf.Matrix4d, and
    the matrix from the object's parent to that pose) for ``prim`` relative to the default prim."""
    from pxr import Gf, UsdGeom
    top = stage.GetDefaultPrim()
    target = stage.GetPrimAtPath(top.GetPath().AppendPath(prim)) if top else None
    if not target:
        raise ValueError(f'{prim}: no such prim below the default prim')
    cache = UsdGeom.XformCache()
    scale = Gf.Matrix4d().SetScale(Gf.Vec3d(*_composed_top_scale(top)))
    placed = cache.GetLocalToWorldTransform(target) * cache.GetLocalToWorldTransform(top).GetInverse() * scale
    pose = Gf.Matrix4d().SetRotate(placed.RemoveScaleShear().ExtractRotationQuat()) \
        .SetTranslateOnly(placed.ExtractTranslation())
    local = UsdGeom.Xformable(target).GetLocalTransformation()
    return target, pose, local.GetInverse() * placed * pose.GetInverse()


@functools.lru_cache(maxsize=8)
def _scene_stage(path, _mtime):
    from pxr import Usd
    return Usd.Stage.Open(path)


def object_pose(path, prim):
    """``{'position', 'orientation'}`` (m, wxyz) of the scene object ``prim`` (relative to the default
    prim) in the frame a ``usd_object`` of ``path`` places at its entity pose: where the entity of
    ``usd_object(path, prim=prim)`` goes to keep the object in place."""
    path = Path(path).resolve()
    stage = _scene_stage(str(path), path.stat().st_mtime_ns)
    _, pose, _ = _object_placement(stage, str(prim).strip('/'))
    q = pose.ExtractRotationQuat()
    return {'position': [round(v, 6) for v in pose.ExtractTranslation()],
            'orientation': [round(q.GetReal(), 8), *(round(v, 8) for v in q.GetImaginary())]}


def object_layer(path, prim, convex=False):
    """A cached USD layer holding only the scene object ``prim`` of ``path`` (relative to its default prim),
    for ``usd_object(prim=...)``: an object taken out of an imported scene (a chair of a house) to be
    moved, simulated or used by the Task. Its origin is the object's own pose (``object_pose``); units and
    the transforms of the scene above it are on an inner prim, since Isaac Lab replaces the top prim's
    transform. The simplified collider the environment wrapper wrote beside a gprim object comes along.
    ``convex`` approximates its triangle-mesh colliders by convex hulls, which PhysX needs on a dynamic
    body. Returns the layer path."""
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    path, prim = Path(path).resolve(), str(prim).strip('/')
    if not prim or '..' in prim.split('/'):
        raise ValueError(f'{workspace_path(path)}: prim is a path relative to the default prim')
    key = hashlib.sha256(json.dumps({'source': source_hashes(path), 'prim': prim, 'convex': bool(convex),
                                     'writer': digest(__file__)}, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / '.cache/usd_objects' / f'{path.stem}_{prim.rpartition("/")[2][:40]}_{key}.usda'
    if output.is_file():
        return output
    stage = Usd.Stage.Open(str(path))
    try:
        target, _, frame = _object_placement(stage, prim)
    except ValueError as error:
        raise ValueError(f'{workspace_path(path)}: {error}') from None
    beside = stage.GetPrimAtPath(target.GetPath().GetParentPath().AppendChild(target.GetName() + COLLIDER_SUFFIX))
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateAnonymous('.usda')
    out = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)
    out.SetDefaultPrim(UsdGeom.Xform.Define(out, '/Object').GetPrim())
    UsdGeom.Xform.Define(out, '/Object/frame').AddTransformOp().Set(frame)
    # Absolute while composing here (an anonymous layer resolves nothing relative), relative in the file.
    holders = {'/Object/frame/object': target.GetPath()}
    if beside:
        holders['/Object/frame/collider'] = beside.GetPath()
    for holder, source in holders.items():
        out.DefinePrim(holder).GetReferences().AddReference(str(path), source)
    if convex:
        for part in Usd.PrimRange(out.GetPrimAtPath('/Object')):
            if part.IsA(UsdGeom.Mesh) and part.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.MeshCollisionAPI.Apply(out.OverridePrim(part.GetPath())).CreateApproximationAttr(
                    UsdPhysics.Tokens.convexHull)
    del out, stage, target, beside
    for holder, source in holders.items():
        layer.GetPrimAtPath(holder).referenceList.prependedItems = [
            Sdf.Reference(os.path.relpath(path, output.parent), source)]
    staging = output.with_suffix(f'.{os.getpid()}.usda')
    layer.Export(str(staging))
    staging.replace(output)
    return output


def joint_spring_layer(path, springs):
    """A cached USD layer over ``path`` whose named joints get these drive springs (``usd_object.springs``).

    ``springs`` maps joint names to ``{stiffness, damping, target}`` in SI units (N/m, N s/m, m for
    prismatic joints; N m/rad, N m s/rad, rad for revolute joints). The joint's own UsdPhysics drive is
    edited (or added, force type, unlimited force) and its target is always written: ``target``, default
    0 (the asset's rest pose), within the joint limits. SimReady buttons author their target just beyond
    the rest limit as a preload, which Newton replaces with the middle of the range (ModelBuilder:
    out-of-limit targets), leaving them half pressed, and their stiffness is too weak to hold the cap
    up. Returns the layer path."""
    import math
    from pxr import Sdf, Usd, UsdPhysics
    path = Path(path).resolve()
    if not isinstance(springs, dict) or not springs:
        raise ValueError(f'{workspace_path(path)}: springs maps joint names to {{stiffness, damping, target}}')
    stage = Usd.Stage.Open(str(path))
    joints = {prim.GetName(): prim for prim in stage.Traverse()
              if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)}
    edits = {}
    for name, spring in springs.items():
        if name not in joints:
            raise ValueError(f'{workspace_path(path)}: springs names unknown joint {name!r}; movable joints: '
                             f'{", ".join(sorted(joints))}')
        if not isinstance(spring, dict) or set(spring) - {'stiffness', 'damping', 'target'} or 'stiffness' not in spring:
            raise ValueError(f'{workspace_path(path)}: springs.{name} is {{stiffness, damping, target}}')
        values = {key: spring.get(key, None if key == 'stiffness' else 0.) for key in ('stiffness', 'damping', 'target')}
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (
                    key != 'target' and value < 0):
                raise ValueError(f'{workspace_path(path)}: springs.{name}.{key} must be a finite '
                                 f'{"" if key == "target" else "non-negative "}number')
        edits[name] = {key: float(value) for key, value in values.items()}
        joint = UsdPhysics.RevoluteJoint(joints[name]) if joints[name].IsA(UsdPhysics.RevoluteJoint) else \
            UsdPhysics.PrismaticJoint(joints[name])
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        target = math.degrees(edits[name]['target']) if joints[name].IsA(UsdPhysics.RevoluteJoint) else edits[name]['target']
        if lower is not None and upper is not None and math.isfinite(lower) and math.isfinite(upper) and \
                not lower - 1e-9 <= target <= upper + 1e-9:
            raise ValueError(f'{workspace_path(path)}: springs.{name}.target {edits[name]["target"]} lies outside the '
                             f'joint limits [{lower}, {upper}]{" deg" if joints[name].IsA(UsdPhysics.RevoluteJoint) else ""}')
    key = hashlib.sha256(json.dumps({'source': source_hashes(path), 'springs': edits,
                                     'writer': digest(__file__)}, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / '.cache/usd_springs' / f'{path.stem}_{key}.usda'
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateAnonymous('.usda')
    out = Usd.Stage.Open(layer)
    # Absolute while edited in memory (an anonymous layer has no directory), relative once written.
    out.GetRootLayer().subLayerPaths.append(str(path))
    if stage.GetDefaultPrim():
        out.SetDefaultPrim(out.OverridePrim(stage.GetDefaultPrim().GetPath()))
    for name, edit in edits.items():
        prim = joints[name]
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        token = 'angular' if angular else 'linear'
        scale = math.pi / 180 if angular else 1.  # UsdPhysics angular drives are per degree
        drive = UsdPhysics.DriveAPI.Apply(_editable(out, prim, stage.GetPseudoRoot()), token)
        drive.CreateStiffnessAttr(edit['stiffness'] * scale)
        drive.CreateDampingAttr(edit['damping'] * scale)
        drive.CreateTargetPositionAttr(math.degrees(edit['target']) if angular else edit['target'])
        if not prim.HasAPI(UsdPhysics.DriveAPI, token):
            drive.CreateTypeAttr('force')
            drive.CreateMaxForceAttr(float('inf'))
    del out, stage, joints  # no stage recomposes the layer once its path turns relative
    layer.subLayerPaths[0] = os.path.relpath(path, output.parent)
    staging = output.with_suffix(f'.{os.getpid()}.usda')
    layer.Export(str(staging))
    staging.replace(output)
    return output


# SimReady USD -> wrapper USD ---------------------------------------------------

def _rigid_body_of(prim):
    from pxr import UsdPhysics
    while prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim = prim.GetParent()
    return prim if prim else None


def usd_authored_mass(path):
    """Whether the USD at `path` authors a mass or density on any rigid body or collider."""
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(Path(path).resolve()))
    return any(prim.HasAPI(UsdPhysics.MassAPI) and (UsdPhysics.MassAPI(prim).GetMassAttr().Get()
                                                   or UsdPhysics.MassAPI(prim).GetDensityAttr().Get())
               for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))


def usd_bounds(path):
    """``(minimum, maximum)`` corners (m) of the drawn geometry of a USD file's default prim, in that
    prim's own frame (the frame a spawned asset is placed by), or None without drawable geometry. Cheap:
    authored extents, no simulator; editors size a placeholder with it while the native mesh is made. A
    static environment (``run.py asset environment``) gives its recorded collider bounds: its drawn ones may
    include a sky dome kilometres wide."""
    path = Path(path).resolve()
    record = path.parent.parent / 'ASSET.json'
    if record.is_file():
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            low, high = json.loads(record.read_text())['environment']['collider_bounds']
            if all(math.isfinite(value) and abs(value) < 1e6 for value in low + high) and \
                    all(a <= b for a, b in zip(low, high)):
                return list(low), list(high)
    return _usd_bounds(str(path), path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _usd_bounds(path, _mtime):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(path)
    prim = stage.GetDefaultPrim() or next(iter(stage.GetPseudoRoot().GetChildren()), None)
    if not prim:
        return None
    box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                            useExtentsHint=True).ComputeUntransformedBound(prim).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    unit = UsdGeom.GetStageMetersPerUnit(stage) or 1.
    return [float(v) * unit for v in box.GetMin()], [float(v) * unit for v in box.GetMax()]


def usd_body_structure(path):
    """What the USD at `path` simulates, read from its UsdPhysics schemas with the default variants:
    'articulation' (rigid bodies joined by joints, with an articulation root), 'rigid' (one rigid
    body) or 'static' (no rigid body: colliders and visuals only, e.g. terrain tiles)."""
    path = Path(path).resolve()
    return _usd_body_structure(str(path), path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _usd_body_structure(path, _mtime):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(path)
    prims = list(Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()))
    # physics:rigidBodyEnabled = false makes a body static (scene props in Isaac environments).
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)
              and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False]
    joints = [prim for prim in prims if prim.IsA(UsdPhysics.Joint)
              and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False
              and any(UsdPhysics.Joint(prim).GetBody0Rel().GetTargets() + UsdPhysics.Joint(prim).GetBody1Rel().GetTargets())]
    if joints or any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in prims):
        if not any(prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in prims):
            raise ValueError(f'{workspace_path(path)}: jointed bodies without ArticulationRootAPI; wrap SimReady '
                             'assets with: run.py asset simready <source> assets/<name>')
        return 'articulation'
    if len(bodies) > 1:
        raise ValueError(f'{workspace_path(path)}: {len(bodies)} rigid bodies without joints; one USD object '
                         'is one rigid body or one articulation')
    return 'rigid' if bodies else 'static'


def simready_layer(source, target):
    """Write `target`, a USD layer that references the SimReady `source` unchanged and adds overrides:

    - rigid bodies and colliders inside instances become ordinary prims: Isaac Lab authors physics
      properties (fixed root, mass, material) through apply_nested, which skips instance proxies with a
      warning, and OvPhysX ignores the mass authored on a collider inside an instance (a 0.157 kg
      SimReady apple weighed 0.087 kg);
    - joint body0/body1 targets naming a collider mesh are retargeted to the rigid body containing it,
      only where the mesh frame equals the body frame, so the authored joint frames keep their meaning;
    - an articulated asset without ArticulationRootAPI, or with it on a prim that is no rigid body, gets it
      on the one body no joint moves;
    - the root body (articulation root or single rigid body) takes the asset frame (``_reroot``): Isaac Lab
      places that body where the scene places the asset;
    - revolute and prismatic limits are made hard in Newton (newton:limitStiffness = inf). UsdPhysics
      limits are hard stops in PhysX; Newton otherwise imports them as a 1e4 spring with almost no
      damping, which lets a 5 kg SimReady drawer pass its limit by ~0.1 m and bounce back;
    - an asset without revolute or prismatic joints becomes one rigid body on the default prim, so
      every object is dynamic: parts held by fixed joints or by nothing (a jar and its lid) are merged
      (their joints deactivated, their summed mass authored on the new body; an articulation of fixed
      joints only has no degrees of freedom, which Isaac Lab's OvPhysX articulation cannot hold), and
      colliders without a rigid body get one.
    Returns what was changed."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    source, target = Path(source).resolve(), Path(target).resolve()
    stage = Usd.Stage.Open(str(source))
    top = stage.GetDefaultPrim()
    if not top:
        raise ValueError(f'{source}: no default prim')
    target.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(target))
    out = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(out, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(out, UsdGeom.GetStageMetersPerUnit(stage))
    wrapper = out.DefinePrim(top.GetPath(), top.GetTypeName() or 'Xform')
    wrapper.GetReferences().AddReference(os.path.relpath(source, target.parent))
    out.SetDefaultPrim(wrapper)
    report = {'uninstanced': [], 'retargeted': [], 'articulation_root': None, 'hard_limits': []}
    bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [prim for prim in Usd.PrimRange(top, Usd.TraverseInstanceProxies())
                 if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsInstanceProxy()]
    for prim in bodies + colliders:
        for prefix in prim.GetPath().GetPrefixes():
            if stage.GetPrimAtPath(prefix).IsInstanceable() and str(prefix) not in report['uninstanced']:
                out.OverridePrim(prefix).SetInstanceable(False)
                report['uninstanced'].append(str(prefix))
    joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
    if not any(prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint) for prim in joints):
        report.update(_simready_single_body(stage, out, wrapper, joints))
        report.update(_reroot(out, wrapper, out.GetPrimAtPath(report['rigid_body'])))
        layer.Save()
        return report
    cache = UsdGeom.XformCache()
    moved = set()
    group = {body.GetPath(): body.GetPath() for body in bodies}  # union-find over the jointed bodies

    def root(path):
        while group[path] != path:
            path = group[path]
        return path
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        ends = []
        for index, relationship in enumerate((joint.GetBody0Rel(), joint.GetBody1Rel())):
            targets = relationship.GetTargets()
            if not targets:
                continue
            end = stage.GetPrimAtPath(targets[0])
            body = _rigid_body_of(end)
            if body is None:
                raise ValueError(f'{prim.GetPath()}: body{index} {targets[0]} is not inside a rigid body')
            ends.append(body.GetPath())
            if index == 1:
                moved.add(body.GetPath())
            if body == end:
                continue
            offset = cache.GetLocalToWorldTransform(end) * cache.GetLocalToWorldTransform(body).GetInverse()
            if not Gf.IsClose(offset, Gf.Matrix4d(1), 1e-6):
                raise ValueError(f'{prim.GetPath()}: body{index} {end.GetPath()} is not at the frame of its rigid '
                                 f'body {body.GetPath()}; retargeting would move the joint frame')
            override = UsdPhysics.Joint(out.OverridePrim(prim.GetPath()))
            (override.GetBody0Rel() if index == 0 else override.GetBody1Rel()).SetTargets([body.GetPath()])
            report['retargeted'].append(f'{prim.GetName()}.body{index}')
        if len(ends) == 2:
            group[root(ends[0])] = root(ends[1])
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            out.OverridePrim(prim.GetPath()).CreateAttribute(
                'newton:limitStiffness', Sdf.ValueTypeNames.Float).Set(float('inf'))
            report['hard_limits'].append(prim.GetName())
    authored = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if joints and not any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in authored):
        # A root authored on a plain Xform above the bodies (e.g. the default prim) leaves Isaac Lab no
        # link to fix (fix_root_link raises): it moves to the base body, like a missing root.
        for prim in authored:
            _editable(out, prim, top).RemoveAPI(UsdPhysics.ArticulationRootAPI)
        bases = [body for body in bodies if body.GetPath() not in moved]
        parts = {root(body.GetPath()) for body in bodies}
        if len(parts) > 1:
            raise ValueError(f'{source}: {len(parts)} separate parts that no joint connects (e.g. a box and its '
                             f'loose lid), with bases {[str(body.GetPath()) for body in bases]}; one USD object is '
                             'one rigid body or one articulation, so this asset is not supported')
        if len(bases) != 1:
            raise ValueError(f'{source}: expected one rigid body that no joint moves, found '
                             f'{[str(body.GetPath()) for body in bases]}')
        UsdPhysics.ArticulationRootAPI.Apply(out.OverridePrim(bases[0].GetPath()))
        report['articulation_root'] = str(bases[0].GetPath())
    root = next(prim for prim in Usd.PrimRange(out.GetPrimAtPath(top.GetPath()), Usd.TraverseInstanceProxies())
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI))
    report.update(_reroot(out, wrapper, root))
    layer.Save()
    return report


def _transform(prim, matrix):
    """Author ``matrix`` as the prim's whole local transform (one op, in the wrapper layer)."""
    from pxr import UsdGeom
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(matrix)


def _reroot(out, top, root):
    """Put the root body's frame on the default prim (the asset frame), keeping every world pose.

    Isaac Lab places an asset by writing its root body pose: a root body away from the asset origin
    (a SimReady ladder's frame is 2.2 m up, a cardboard box's 0.41 m) would start that far below where
    the scene puts it and be pushed out of the ground. The root body's children take its old transform;
    joint frames and mass properties authored in its frame are re-expressed."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    origin = cache.GetLocalToWorldTransform(top).GetInverse()
    relative = cache.GetLocalToWorldTransform(root) * origin
    if Gf.IsClose(relative, Gf.Matrix4d(1), 1e-9):
        return {}
    if not Gf.IsClose(Gf.Transform(relative).GetScale(), Gf.Vec3d(1, 1, 1), 1e-6):
        raise ValueError(f'{root.GetPath()}: the root body is scaled relative to the asset; cannot move its frame')
    for child in root.GetChildren():
        stack = [child]
        while stack:
            prim = stack.pop()
            if prim.IsA(UsdGeom.Xformable):
                local = UsdGeom.Xformable(prim).GetLocalTransformation()
                _transform(_editable(out, prim, top), local * relative)
            else:  # e.g. a Scope: its children inherit the body frame directly
                stack.extend(prim.GetChildren())
    parent = cache.GetLocalToWorldTransform(root.GetParent()) * origin
    _transform(_editable(out, root, top), parent.GetInverse())
    for prim in out.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        for relationship, position, rotation in ((joint.GetBody0Rel(), joint.GetLocalPos0Attr(), joint.GetLocalRot0Attr()),
                                                 (joint.GetBody1Rel(), joint.GetLocalPos1Attr(), joint.GetLocalRot1Attr())):
            if relationship.GetTargets()[:1] != [root.GetPath()]:
                continue
            frame = Gf.Matrix4d().SetRotate(Gf.Quatd(rotation.Get() or Gf.Quatf(1))) * \
                Gf.Matrix4d().SetTranslate(Gf.Vec3d(position.Get() or Gf.Vec3f(0))) * relative
            override = UsdPhysics.Joint(out.OverridePrim(prim.GetPath()))
            target = (override.GetLocalPos0Attr(), override.GetLocalRot0Attr()) if relationship == joint.GetBody0Rel() \
                else (override.GetLocalPos1Attr(), override.GetLocalRot1Attr())
            target[0].Set(Gf.Vec3f(frame.ExtractTranslation()))
            target[1].Set(Gf.Quatf(frame.ExtractRotationQuat()))
    if root.HasAPI(UsdPhysics.MassAPI):
        mass, editable = UsdPhysics.MassAPI(root), UsdPhysics.MassAPI(_editable(out, root, top))
        centre = mass.GetCenterOfMassAttr().Get()
        if centre is not None and all(abs(v) < 1e30 for v in centre):
            editable.GetCenterOfMassAttr().Set(Gf.Vec3f(relative.Transform(Gf.Vec3d(centre))))
        axes = mass.GetPrincipalAxesAttr().Get()
        if axes is not None and mass.GetPrincipalAxesAttr().HasAuthoredValue():
            turned = Gf.Matrix4d().SetRotate(Gf.Quatd(axes)) * Gf.Matrix4d().SetRotate(relative.ExtractRotationQuat())
            editable.GetPrincipalAxesAttr().Set(Gf.Quatf(turned.ExtractRotationQuat()))
    return {'rerooted': {'body': str(root.GetPath()), 'offset': [round(v, 6) for v in relative.ExtractTranslation()]}}


def usd_root_offset(path):
    """Translation (m) of an asset's root body (articulation root, else its single rigid body) from the asset
    frame, or None when they coincide or there is no body. Isaac Lab places that body where the scene places
    the asset; ``simready_layer`` puts them together."""
    path = Path(path).resolve()
    return _usd_root_offset(str(path), path.stat().st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _usd_root_offset(path, _mtime):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(path)
    top = stage.GetDefaultPrim()
    if not top:
        return None
    prims = list(Usd.PrimRange(top, Usd.TraverseInstanceProxies()))
    roots = [prim for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    root = roots[0] if roots else bodies[0] if len(bodies) == 1 else None
    if root is None:
        return None
    cache = UsdGeom.XformCache()
    relative = cache.GetLocalToWorldTransform(root) * cache.GetLocalToWorldTransform(top).GetInverse()
    if Gf.IsClose(relative, Gf.Matrix4d(1), 1e-6):
        return None
    return [float(v) * (UsdGeom.GetStageMetersPerUnit(stage) or 1.) for v in relative.ExtractTranslation()]


def _editable(out, prim, root):
    """Override ``prim`` in ``out`` after making its instanceable ancestors below ``root`` ordinary prims."""
    for prefix in prim.GetPath().GetPrefixes():
        if prefix.HasPrefix(root.GetPath()) and prefix != prim.GetPath() and prim.GetStage().GetPrimAtPath(prefix).IsInstanceable():
            out.OverridePrim(prefix).SetInstanceable(False)
    return out.OverridePrim(prim.GetPath())


def _simready_single_body(stage, out, wrapper, joints):
    """Make the default prim of a SimReady asset without movable joints its one rigid body (simready_layer)."""
    from pxr import Usd, UsdPhysics
    top = stage.GetDefaultPrim()
    prims = list(Usd.PrimRange(top, Usd.TraverseInstanceProxies()))
    bodies = [prim for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    colliders = [prim for prim in prims if prim.HasAPI(UsdPhysics.CollisionAPI)]
    report = {'merged_bodies': [], 'deactivated_joints': [], 'rigid_body': None, 'mass': None}
    for prim in joints:
        out.OverridePrim(prim.GetPath()).SetActive(False)
        report['deactivated_joints'].append(prim.GetName())
    for prim in prims:
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) and prim != top:
            _editable(out, prim, top).RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if len(bodies) == 1 and not joints:
        report['rigid_body'] = str(bodies[0].GetPath())  # already one rigid body, kept where it is authored
        return report
    masses = []
    for body in bodies:
        mass = UsdPhysics.MassAPI(body).GetMassAttr().Get() if body.HasAPI(UsdPhysics.MassAPI) else None
        if not mass:
            parts = [UsdPhysics.MassAPI(prim).GetMassAttr().Get() for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.HasAPI(UsdPhysics.MassAPI)]
            mass = sum(parts) if parts and all(parts) else None
        masses.append(mass)
        if body != top:
            override = _editable(out, body, top)
            override.RemoveAPI(UsdPhysics.RigidBodyAPI)
            override.RemoveAPI(UsdPhysics.MassAPI)
            report['merged_bodies'].append(str(body.GetPath()))
    if not colliders:
        raise ValueError(f'{top.GetPath()}: no collider to simulate')
    UsdPhysics.RigidBodyAPI.Apply(wrapper)
    report['rigid_body'] = str(top.GetPath())
    if masses and all(masses):
        report['mass'] = float(sum(masses))
        UsdPhysics.MassAPI.Apply(wrapper).CreateMassAttr(report['mass'])
    return report


def simready_package(source):
    """SimReady packaging metadata and mirror revision for a source inside <asset>/simready_usd/."""
    source = Path(source).resolve()
    package = source.parent.parent / 'com.nvidia.simready.packaging.json'
    if source.parent.name != 'simready_usd' or not package.is_file():
        raise ValueError(f'{source}: expected <asset>/simready_usd/<file> beside com.nvidia.simready.packaging.json')
    metadata = json.loads(package.read_text())
    record = {'asset': source.parent.parent.name, 'package_id': metadata['package_id'],
              'license': metadata['license'], 'packaging': workspace_path(package)}
    mirror = source.parent.parent.parent / '.mirror' / 'report.json'
    if mirror.is_file():
        report = json.loads(mirror.read_text())
        record.update(repository=report['url'], revision=report['revision'])
    else:  # a single asset downloaded by shared.simready: its pinned dataset
        from shared.simready import SOURCES
        pinned = SOURCES.get(source.parent.parent.parent.name)
        if pinned:
            record.update(repository=f"https://huggingface.co/datasets/{pinned['repo']}", revision=pinned['revision'])
    return record


def simready(source, output, name=None):
    """Write OUTPUT/usd/NAME.usda wrapping a SimReady USD, and ASSET.json (no MJCF form)."""
    source, output = Path(source).resolve(), Path(output).resolve()
    package = simready_package(source)
    name = name or package['asset']
    if (output / 'ASSET.json').exists() or (output / 'usd').exists():
        raise FileExistsError(f'{output} already has ASSET.json or usd/; remove them or choose another output')
    from pxr import UsdUtils
    missing = UsdUtils.ComputeAllDependencies(str(source))[2]
    if missing:
        raise FileNotFoundError(f'{source}: {len(missing)} referenced files are missing, e.g. {missing[0]}; '
                                'complete the mirror by rerunning tools/simready_mirror.py')
    target = output / 'usd' / f'{name}.usda'
    report = simready_layer(source, target)
    record = {
        'source': workspace_path(source), 'source_format': 'usd', 'source_sha256': digest(source),
        'source_layers_sha256': usd_layers(source), 'simready': package,
        'formats': {'usd': workspace_path(target)},
        'steps': [{'from': 'usd', 'to': 'usd', 'tool': 'shared.asset_formats.simready_layer', **report}],
        'joint_order': usd_joint_order(target), 'alignment': None,
        'sha256': {target.relative_to(output).as_posix(): digest(target)},
    }
    (output / 'ASSET.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def _mesh_has_geometry(prim):
    from pxr import UsdGeom
    mesh = UsdGeom.Mesh(prim)
    return bool(mesh.GetPointsAttr().Get() and mesh.GetFaceVertexIndicesAttr().Get()
                and mesh.GetFaceVertexCountsAttr().Get())


STUB_MARK = 'emptyStubForMissingUpstreamFile'


def _write_stub(path, scene, replacement=None):
    """Write the layer standing in for a missing upstream file: empty, or referencing `replacement`
    rescaled to the units and up axis of the `scene` stage."""
    from pxr import Gf, Sdf, Usd, UsdGeom
    path.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.FindOrOpen(str(path)) if path.exists() else Sdf.Layer.CreateNew(str(path))
    layer.Clear()
    stage = Usd.Stage.Open(layer)
    units, up = UsdGeom.GetStageMetersPerUnit(scene), UsdGeom.GetStageUpAxis(scene)
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, units)
    prim = stage.DefinePrim('/missing', 'Xform')
    stage.SetDefaultPrim(prim)
    data = {STUB_MARK: True}
    if replacement is not None:
        data['substitute'] = os.path.relpath(replacement, path.parent)
        # A child prim, since the referencing prim's own xformOpOrder overrides ops authored on the root.
        child = stage.DefinePrim('/missing/substitute', 'Xform')
        child.GetReferences().AddReference(data['substitute'])
        other = Usd.Stage.Open(str(replacement))
        xform = UsdGeom.Xformable(child)
        if UsdGeom.GetStageUpAxis(other) != up:
            xform.AddRotateXOp(opSuffix='upAxis').Set(90.0 if up == UsdGeom.Tokens.z else -90.0)
        if (ratio := UsdGeom.GetStageMetersPerUnit(other) / units) != 1.0:
            xform.AddScaleOp(opSuffix='unitsResolve').Set(Gf.Vec3f(ratio))
    layer.customLayerData = data
    layer.Save()


def _footprint_boxes(triangles, cell=0.2, levels=(0.3, 0.9, 1.5)):
    """Boxes covering where `triangles` ([T, 3, 3]) cross the planes `levels` above their lowest point,
    rasterized on a `cell` grid turned to the dominant edge direction, enclosed space filled and runs
    merged into rectangles: [(centre_x, centre_y, size_x, size_y)], the grid yaw (radians), bottom, top."""
    import numpy as np
    from scipy import ndimage
    bottom, top = float(triangles[..., 2].min()), float(triangles[..., 2].max())
    starts, ends = [], []
    for level in levels:
        if bottom + level >= top:
            break
        height = triangles[..., 2] - (bottom + level)
        crossing = (height.min(1) < 0) & (height.max(1) > 0)
        corners, height = triangles[crossing], height[crossing]
        found, points = [], []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            edge = (height[:, i] < 0) != (height[:, j] < 0)
            t = (height[edge, i] / (height[edge, i] - height[edge, j]))[:, None]
            found.append(np.nonzero(edge)[0])
            points.append(corners[edge, i, :2] + t * (corners[edge, j, :2] - corners[edge, i, :2]))
        # a triangle straddling the plane changes sign on exactly two edges
        points = np.concatenate(points)[np.argsort(np.concatenate(found), kind='stable')]
        starts.append(points[0::2])
        ends.append(points[1::2])
    if not starts or not sum(map(len, starts)):
        return [], 0.0, bottom, top
    start, end = np.concatenate(starts), np.concatenate(ends)
    delta = end - start
    degrees = np.round(np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))).astype(int) % 90
    yaw = np.radians(np.argmax(np.bincount(degrees, weights=np.linalg.norm(delta, axis=1), minlength=90)))
    turn = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    start, end = start @ turn, end @ turn
    low = np.minimum(start.min(0), end.min(0))
    shape = np.floor((np.maximum(start.max(0), end.max(0)) - low) / cell).astype(int) + 1
    count = np.ceil(np.linalg.norm(end - start, axis=1) / (cell / 2)).astype(int) + 1
    segment = np.repeat(np.arange(len(start)), count)
    t = (np.arange(len(segment)) - np.repeat(np.cumsum(count) - count, count)) / np.repeat(
        np.maximum(count - 1, 1), count)
    samples = start[segment] + t[:, None] * (end - start)[segment]
    index = np.clip(np.floor((samples - low) / cell).astype(int), 0, shape - 1)
    grid = np.zeros(shape, bool)
    grid[index[:, 0], index[:, 1]] = True
    grid = ndimage.binary_fill_holes(grid)
    boxes, open_runs = [], {}
    for x in range(shape[0] + 1):
        column = np.r_[False, grid[x], False] if x < shape[0] else np.zeros(shape[1] + 2, bool)
        edges = np.nonzero(column[1:] != column[:-1])[0]
        runs = set(zip(edges[0::2], edges[1::2]))
        for run in [run for run in open_runs if run not in runs]:
            first = open_runs.pop(run)
            centre = low + cell * np.array([(first + x) / 2, (run[0] + run[1]) / 2])
            boxes.append((*(turn @ centre), (x - first) * cell, (run[1] - run[0]) * cell))
        for run in runs:
            open_runs.setdefault(run, x)
    return boxes, float(yaw), bottom, top


def _mesh_triangles(prim, matrix):
    """(points [N, 3] transformed by `matrix`, fan-triangulated faces [T, 3]) of a Mesh prim."""
    import numpy as np
    from pxr import UsdGeom
    mesh, matrix = UsdGeom.Mesh(prim), np.array(matrix)
    points = np.array(mesh.GetPointsAttr().Get(), float) @ matrix[:3, :3] + matrix[3, :3]
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    fans = np.maximum(counts - 2, 0)
    first = np.repeat(np.cumsum(counts) - counts, fans)
    step = np.arange(fans.sum()) - np.repeat(np.cumsum(fans) - fans, fans)
    return points, indices[np.stack([first, first + step + 1, first + step + 2], 1)]


def _simplify_colliders(out, top, cell, editable, units, tile=8.0):
    """Replace every collider Mesh below `top` by its vertices snapped to a `cell` m grid (degenerate and
    repeated triangles dropped), merged into one invisible collider per `tile` m square: Newton holds static
    triangle meshes per shape and per triangle, and a furnished GRScenes home has 4400 collider meshes of
    35 M triangles (beyond 20 GB at MuJoCo conversion). The original meshes stay as visuals. Geometry is in
    the frame of `top`, whose length unit is `units` m.

    Colliders of one scene object (shared/assembly/composition.py: a wall, a cabinet) stay with it, as one
    simplified mesh below the object's prim, so removing the object (``usd_object.exclude``) removes its
    collision too; only colliders outside every object are merged into tiles."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    report = {'cell': cell, 'tile': tile}
    cell, tile = cell / units, tile / units
    cache = UsdGeom.XformCache()
    into_top = cache.GetLocalToWorldTransform(out.GetPrimAtPath(top.GetPath())).GetInverse()
    footprints = top.GetPath().AppendChild('footprint_colliders')
    from shared.assembly.composition import object_of, objects_below
    objects = objects_below(out.GetPrimAtPath(top.GetPath()))
    tiles, owned, found, before = {}, {}, [], 0
    for prim in Usd.PrimRange(out.GetPrimAtPath(top.GetPath()), Usd.TraverseInstanceProxies()):
        if prim.GetPath().HasPrefix(footprints) or not prim.IsA(UsdGeom.Mesh) or \
                not prim.HasAPI(UsdPhysics.CollisionAPI) or not _mesh_has_geometry(prim):
            continue
        points, triangles = _mesh_triangles(prim, cache.GetLocalToWorldTransform(prim) * into_top)
        before += len(triangles)
        found.append(prim.GetPath())
        cells, inverse = np.unique(np.round(points / cell).astype(np.int64), axis=0, return_inverse=True)
        triangles = inverse.reshape(-1)[triangles]
        triangles = triangles[(triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
                              & (triangles[:, 0] != triangles[:, 2])]
        triangles = triangles[np.unique(np.sort(triangles, axis=1), axis=0, return_index=True)[1]]
        if not len(triangles):
            continue
        vertices = cells * cell
        owner = object_of(str(prim.GetPath().MakeRelativePath(top.GetPath())), objects)
        if owner is not None:
            owned.setdefault(owner, []).append((vertices, triangles))
            continue
        keys = np.floor(vertices[triangles].mean(1)[:, :2] / tile).astype(np.int64)
        for key in np.unique(keys, axis=0):
            tiles.setdefault(tuple(key), []).append((vertices, triangles[np.all(keys == key, axis=1)]))
    for path in found:  # collected first: uninstancing while traversing invalidates the instance proxies
        prim = editable(path)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
    after = 0

    def merged(parts):
        used = [np.unique(triangles) for _, triangles in parts]
        offsets = np.cumsum([0] + [len(index) for index in used])
        points = np.concatenate([vertices[index] for (vertices, _), index in zip(parts, used)])
        faces = np.concatenate([np.searchsorted(index, triangles) + offset
                                for (_, triangles), index, offset in zip(parts, used, offsets)])
        return points, faces

    def collider(mesh, points, faces):
        mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points.tolist()])
        mesh.CreateFaceVertexCountsAttr([3] * len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.reshape(-1).tolist())
        UsdGeom.Imageable(mesh).CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
        return len(faces)
    for name, parts in sorted(owned.items()):
        path = top.GetPath().AppendPath(name)
        editable(path)  # ordinary prims down to the object
        if out.GetPrimAtPath(path).IsInstance():
            out.OverridePrim(path).SetInstanceable(False)
        # Below an Xform object; beside a gprim object (gprims do not nest), removed with it (pruned_layer).
        gprim = out.GetPrimAtPath(path).IsA(UsdGeom.Gprim)
        holder = path.GetParentPath() if gprim else path
        mesh = UsdGeom.Mesh.Define(out, holder.AppendChild(f'{path.name}{COLLIDER_SUFFIX}' if gprim else 'simplified_collider'))
        # Points stay in the frame of `top`: undo the placement of the prim holding the collider.
        fresh = UsdGeom.XformCache()
        to_top = fresh.GetLocalToWorldTransform(out.GetPrimAtPath(holder)) * \
            fresh.GetLocalToWorldTransform(out.GetPrimAtPath(top.GetPath())).GetInverse()
        mesh.AddTransformOp().Set(to_top.GetInverse())
        after += collider(mesh, *merged(parts))
    holder = UsdGeom.Scope.Define(out, top.GetPath().AppendChild('simplified_colliders'))
    UsdGeom.Imageable(holder).CreatePurposeAttr(UsdGeom.Tokens.guide)
    for (x, y), parts in sorted(tiles.items()):
        mesh = UsdGeom.Mesh.Define(out, holder.GetPath().AppendChild(f'tile_{x}_{y}'.replace('-', 'm')))
        after += collider(mesh, *merged(parts))
    return {**report, 'meshes': len(found), 'triangles_before': before, 'triangles_after': after,
            'tiles': len(tiles), 'object_colliders': len(owned)}


def _cluster_triangles(points, triangles, cap):
    """Merge the vertices of a triangle mesh on the finest grid that leaves at most `cap` distinct,
    non-degenerate triangles. Returns (kept triangle indices, their faces over the merged vertices,
    merged vertex positions, one original vertex per merged vertex)."""
    import numpy as np
    low = points.min(0)
    size = max(float((points.max(0) - low).max()), 1e-9)

    def cluster(cell):
        keys = np.floor((points - low) / cell).astype(np.int64)
        span = keys.max(0) + 1
        _, label = np.unique((keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2], return_inverse=True)
        faces = label.reshape(-1)[triangles]
        ok = np.nonzero((faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2]))[0]
        _, first = np.unique(np.sort(faces[ok], axis=1), axis=0, return_index=True)
        return ok[np.sort(first)], label.reshape(-1)

    fine, coarse = size / 2 ** 14, size / 2
    kept, label = cluster(fine)
    if len(kept) > cap:
        for _ in range(14):
            middle = (fine * coarse) ** .5
            found = cluster(middle)
            if len(found[0]) > cap:
                fine = middle
            else:
                coarse, (kept, label) = middle, found
        if len(kept) > cap:
            kept, label = cluster(coarse)
    used, faces = np.unique(label[triangles[kept]], return_inverse=True)
    count = np.bincount(label, minlength=label.max() + 1)[used]
    merged = np.stack([np.bincount(label, points[:, axis], minlength=label.max() + 1)[used] for axis in range(3)], 1)
    first = np.full(label.max() + 1, len(label))
    np.minimum.at(first, label, np.arange(len(label)))
    return kept, faces.reshape(-1, 3), merged / count[:, None], first[used]


def _simplify_visuals(out, top, budget, editable):
    """Bring the visible meshes below `top` to about `budget` triangles in all: every mesh above a common
    cap is clustered to it (_cluster_triangles). Newton loads every visual triangle onto the GPU for its
    viewer and cameras; a furnished GRScenes shop has 37 M, beyond 16 GB. Clustering keeps a subset of the
    original triangles, so face-varying and uniform primvars, normals and GeomSubsets (material bindings)
    carry over; vertex primvars take one merged vertex's value. Colliders are left alone."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, Vt
    meshes = []
    for prim in Usd.PrimRange(out.GetPrimAtPath(top.GetPath()), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI) or not _mesh_has_geometry(prim):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputePurpose() == UsdGeom.Tokens.guide or mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        meshes.append((prim.GetPath(), int(np.maximum(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) - 2, 0).sum())))
    sizes = np.array([size for _, size in meshes] or [0])
    report = {'budget': budget, 'meshes': len(meshes), 'triangles_before': int(sizes.sum())}
    low, high = 1, int(sizes.max())
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if np.minimum(sizes, middle).sum() <= budget else (low, middle - 1)
    cap, after, simplified = low, 0, 0
    for path, size in meshes:
        if size <= cap:
            after += size
            continue
        mesh = UsdGeom.Mesh(out.GetPrimAtPath(path))
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        fans = np.maximum(counts - 2, 0)
        first = np.repeat(np.cumsum(counts) - counts, fans)
        step = np.arange(fans.sum()) - np.repeat(np.cumsum(fans) - fans, fans)
        corners = np.stack([first, first + step + 1, first + step + 2], 1)
        face = np.repeat(np.arange(len(counts)), fans)
        points = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
        kept, faces, merged, vertex = _cluster_triangles(points, indices[corners], cap)
        select = {UsdGeom.Tokens.vertex: vertex, UsdGeom.Tokens.varying: vertex,
                  UsdGeom.Tokens.faceVarying: corners[kept].reshape(-1), UsdGeom.Tokens.uniform: face[kept]}
        expected = {UsdGeom.Tokens.vertex: len(points), UsdGeom.Tokens.varying: len(points),
                    UsdGeom.Tokens.faceVarying: len(indices), UsdGeom.Tokens.uniform: len(counts)}
        target = UsdGeom.Mesh(editable(path))
        target.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(merged.astype(np.float32)))
        target.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
        target.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.reshape(-1).astype(np.int32)))
        target.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(
            np.stack([merged.min(0), merged.max(0)]).astype(np.float32)))
        for name in ('cornerIndices', 'cornerSharpnesses', 'creaseIndices', 'creaseLengths', 'creaseSharpnesses',
                     'holeIndices'):
            if mesh.GetPrim().GetAttribute(name).HasAuthoredValue():
                target.GetPrim().GetAttribute(name).Block()
        values = [(mesh.GetNormalsAttr(), mesh.GetNormalsInterpolation(), mesh.GetNormalsAttr().Get(), None)]
        values += [(primvar.GetAttr(), primvar.GetInterpolation(), primvar.ComputeFlattened(), primvar)
                   for primvar in UsdGeom.PrimvarsAPI(mesh).GetAuthoredPrimvars()]
        for attribute, interpolation, value, primvar in values:
            if value is None or interpolation == UsdGeom.Tokens.constant or not attribute.HasAuthoredValue():
                continue
            override = target.GetPrim().GetAttribute(attribute.GetName())
            if interpolation not in select or len(value) != expected[interpolation]:
                override.Block()
                continue
            array = np.asarray(value)
            type_class = attribute.GetTypeName().type.pythonClass
            chosen = array[select[interpolation]]
            override.Set(type_class.FromNumpy(chosen) if array.dtype != object else type_class(chosen.tolist()))
            if primvar is not None and primvar.IsIndexed():
                UsdGeom.Primvar(override).BlockIndices()
        for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            members = np.asarray(subset.GetIndicesAttr().Get() or [], np.int64)
            UsdGeom.Subset(editable(subset.GetPath())).GetIndicesAttr().Set(
                Vt.IntArray.FromNumpy(np.nonzero(np.isin(face[kept], members))[0].astype(np.int32)))
        after += len(faces)
        simplified += 1
    return {**report, 'cap': cap, 'simplified': simplified, 'triangles_after': after}


def _shrink_textures(out, top, size, editable, folder):
    """Point every image file used below `top` at a copy in `folder` at most `size` pixels on a side.
    Newton's GL viewer uploads a texture with mipmaps per mesh, shared or not: the 20 k meshes of a
    GRScenes shop, each textured at about 600 x 600, would need 36 GB of GPU memory."""
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image
    from pxr import Sdf, Usd
    uses = {}
    for prim in Usd.PrimRange(out.GetPrimAtPath(top.GetPath()), Usd.TraverseInstanceProxies()):
        for attribute in prim.GetAttributes():
            if attribute.GetTypeName() != Sdf.ValueTypeNames.Asset:
                continue
            value = attribute.Get()
            path = value.resolvedPath if value else ''
            if path and Path(path).suffix.lower() in ('.png', '.jpg', '.jpeg', '.tga', '.bmp', '.tif', '.tiff'):
                uses.setdefault(path, []).append((prim.GetPath(), attribute.GetName()))
    folder.mkdir(parents=True, exist_ok=True)

    def shrink(path):
        try:
            with Image.open(path) as image:
                if max(image.size) <= size:
                    return None
                image.thumbnail((size, size))
                name = hashlib.sha1(path.encode()).hexdigest()[:20] + '.png'
                (image if image.mode in ('RGB', 'RGBA', 'L', 'LA') else image.convert('RGBA')).save(folder / name)
                return name
        except OSError:
            return None  # unreadable (e.g. a format PIL lacks): left as it is

    with ThreadPoolExecutor(8) as pool:
        copies = dict(zip(uses, pool.map(shrink, uses)))
    for path, name in copies.items():
        if name is None:
            continue
        for prim_path, attribute in uses[path]:
            editable(prim_path).GetAttribute(attribute).Set(
                Sdf.AssetPath(f'./{folder.name}/{name}'))
    return {'size': size, 'files': len(uses), 'shrunk': sum(name is not None for name in copies.values())}


# Meshes that get no added collider and PointInstancers left as they are (vegetation, sky, decals, effects):
# matched against the prim path; `(?<!s)tree` spares streets.
BARE_NAMES = r'(?i)sky_?(sphere|dome|box)|cloud|decal|grass|weed|leaf|leaves|ivy|foliage|(?<!s)tree|bush|plant|' \
             r'flower|particle'
SMALL_COLLIDER = 0.05  # m: meshes smaller on every axis get no added collider


def _expand_instancers(out, top, source, target, bare, editable):
    """Replace each PointInstancer below `top` by one Xform per instance referencing its prototype from
    `source`: Newton's USD import ignores PointInstancers (Unreal's instanced static meshes: modular
    building pieces, foliage), so their geometry would be neither drawn nor collided with. Instancers whose
    prototypes all match `bare` stay. Copies of a prototype keep one model (composition.model_key)."""
    from pxr import Usd, UsdGeom
    found, kept = [], 0
    walk = iter(Usd.PrimRange(out.GetPrimAtPath(top.GetPath()), Usd.TraverseInstanceProxies()))
    for prim in walk:
        if prim.IsA(UsdGeom.PointInstancer):
            walk.PruneChildren()
            prototypes = UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets()
            if prototypes and all(re.search(bare, str(path)) for path in prototypes):
                kept += 1
            else:
                found.append(prim.GetPath())
    reference = os.path.relpath(source, target.parent)
    instances = missing = 0
    for path in found:
        instancer = UsdGeom.PointInstancer(out.GetPrimAtPath(path))
        prototypes = instancer.GetPrototypesRel().GetTargets()
        if outside := [str(prototype) for prototype in prototypes if not prototype.HasPrefix(top.GetPath())]:
            raise ValueError(f'{path}: prototypes outside {top.GetPath()}: {", ".join(outside)}')
        now = Usd.TimeCode.Default()
        # The prototype's own transform comes with the reference.
        matrices = instancer.ComputeInstanceTransformsAtTime(now, now, UsdGeom.PointInstancer.ExcludeProtoXform,
                                                             UsdGeom.PointInstancer.IgnoreMask)
        shown = instancer.ComputeMaskAtTime(now) or [True] * len(matrices)
        hidden = set(instancer.GetInvisibleIdsAttr().Get() or [])
        ids = instancer.GetIdsAttr().Get() or range(len(matrices))
        visible = instancer.ComputeVisibility() != UsdGeom.Tokens.invisible
        editable(path).SetActive(False)
        if not visible:
            continue
        holder = UsdGeom.Xform.Define(out, path.GetParentPath().AppendChild(f'{path.name}_expanded'))
        holder.AddTransformOp().Set(instancer.GetLocalTransformation())
        purpose = instancer.GetPurposeAttr()
        if purpose.HasAuthoredValue():
            holder.CreatePurposeAttr(purpose.Get())
        for index, (prototype, matrix, on, number) in enumerate(zip(instancer.GetProtoIndicesAttr().Get() or [],
                                                                     matrices, shown, ids)):
            if not on or number in hidden:
                continue
            if not 0 <= prototype < len(prototypes):
                missing += 1
                continue
            copy = UsdGeom.Xform.Define(out, holder.GetPath().AppendChild(f'{prototypes[prototype].name}_{index}'))
            copy.AddTransformOp().Set(matrix)
            out.DefinePrim(copy.GetPath().AppendChild('model')).GetReferences().AddReference(
                reference, prototypes[prototype])
            instances += 1
    return {'instancers': len(found), 'instances': instances, 'kept_instancers': kept,
            'missing_prototypes': missing}


def _add_colliders(out, top, bare, editable):
    """Give every visible mesh below `top` without a collider a triangle-mesh one, except meshes whose path
    matches `bare` or smaller than SMALL_COLLIDER on every axis: Unreal's USD export writes no physics."""
    from pxr import Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    found, skipped, triangles = [], {'bare': 0, 'small': 0}, 0
    walk = iter(Usd.PrimRange(out.GetPrimAtPath(top.GetPath()), Usd.TraverseInstanceProxies()))
    for prim in walk:
        if prim.IsA(UsdGeom.PointInstancer):
            walk.PruneChildren()  # prototypes: Newton does not load them
            continue
        if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.CollisionAPI) or not _mesh_has_geometry(prim):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputePurpose() == UsdGeom.Tokens.guide or mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if re.search(bare, str(prim.GetPath())):
            skipped['bare'] += 1
            continue
        # The wrapper's stage is in metres (the source's units are resolved on `top`).
        if max(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize()) < SMALL_COLLIDER:
            skipped['small'] += 1
            continue
        found.append(prim.GetPath())
        triangles += sum(max(count - 2, 0) for count in mesh.GetFaceVertexCountsAttr().Get())
    for path in found:  # collected first: uninstancing while traversing invalidates the instance proxies
        prim = editable(path)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
    return {'meshes': len(found), 'triangles': triangles, 'skipped': skipped}


def environment_layer(source, target, substitutes=None, footprint_colliders=(), simplify_colliders=None,
                      deactivate=(), simplify_visuals=None, texture_size=None, expand_instancers=False,
                      add_colliders=False, bare=BARE_NAMES):
    """Write `target`, a USD layer that references the scene `source` unchanged and makes it one static
    environment for every simulated world:

    - Newton rejects a stage with composition errors, while Kit only warns (Isaac's public Rivermark
      misses 39 files): a missing file inside the vendored source/ gets an empty stub layer, so only
      the missing part is lost, or one referencing `substitutes[file name]` (a USD path) when given;
      other unresolved arcs deactivate their prim's parent (an inactive
      prim still composes its own arcs, and arcs cannot be deleted across a reference);
    - rigid bodies are removed, joints disabled and articulation roots removed, so props become static
      colliders instead of objects shared by all environments;
    - a Mesh without points or faces becomes an Xform (Newton's mesh import fails on it); a collider
      on it or on any non-Gprim passes to the meshes below that have none, which PhysX collides in
      its place (UsdPhysics defines colliders on Gprims only);
    - meshes scaled to zero are deactivated and mirrored ones re-expressed with a positive scale
      (MuJoCo, which Newton passes every shape to, rejects both);
    - each child of a `footprint_colliders` prim without colliders (e.g. visual-only buildings, too
      detailed for mesh colliders) gets an invisible collider mesh of boxes over its cross-sections
      0.3-1.5 m above its base, from base to top;
    - `expand_instancers` turns PointInstancers into referenced copies (_expand_instancers) and
      `add_colliders` gives meshes without one a triangle-mesh collider (_add_colliders), both sparing
      prims whose path matches `bare` (vegetation, decals): for exports without physics, as Unreal's;
    - `simplify_colliders`, `simplify_visuals` and `texture_size` (_simplify_colliders, _simplify_visuals,
      _shrink_textures) lighten scenes too heavy for Newton; the visual changes go to a binary
      `<target>.visuals.usdc` sublayer, shrunk image copies to `<target stem>.textures/`;
    - a Y-up or non-metre source is rotated and scaled on the wrapper prim to Z-up metres.
    Prims inside instances are uninstanced where an override is needed. Returns what was changed."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    source, target = Path(source).resolve(), Path(target).resolve()
    stage = Usd.Stage.Open(str(source))
    vendored = next((folder for folder in source.parents if (folder / 'SOURCE.json').is_file()), None)
    absent = {Path(found.group(1)) for error in stage.GetCompositionErrors()
              if (found := re.match(r'Could not open asset @([^@]+)@', str(error)))}
    absent = {path for path in absent if vendored and path.resolve().is_relative_to(vendored)}
    absent |= {Path(used.realPath) for used in stage.GetUsedLayers() if used.customLayerData.get(STUB_MARK)}
    substitutes = {name: Path(path).resolve() for name, path in (substitutes or {}).items()}
    if unknown := sorted(set(substitutes) - {path.name for path in absent}):
        raise ValueError(f'substitutes for files that are not missing: {", ".join(unknown)}')
    for path in sorted(absent):
        _write_stub(path, stage, substitutes.get(path.name))
    if absent:
        stage = Usd.Stage.Open(str(source))
    top = stage.GetDefaultPrim()
    if not top:
        raise ValueError(f'{source}: no default prim')
    target.parent.mkdir(parents=True, exist_ok=True)
    layer = Sdf.Layer.CreateNew(str(target))
    out = Usd.Stage.Open(layer)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)
    wrapper = out.DefinePrim(top.GetPath(), top.GetTypeName() or 'Xform')
    wrapper.GetReferences().AddReference(os.path.relpath(source, target.parent))
    out.SetDefaultPrim(wrapper)
    units, up = UsdGeom.GetStageMetersPerUnit(stage), UsdGeom.GetStageUpAxis(stage)
    if units != 1.0 or up != UsdGeom.Tokens.z:
        xform = UsdGeom.Xformable(wrapper)
        xform.ClearXformOpOrder()
        if up == UsdGeom.Tokens.y:
            xform.AddRotateXOp(opSuffix='upAxis').Set(90.0)
        if units != 1.0:
            xform.AddScaleOp(opSuffix='unitsResolve').Set(Gf.Vec3f(units))
    stubs = [used for used in stage.GetUsedLayers() if used.customLayerData.get(STUB_MARK)]
    report = {'stubbed_missing': sorted(workspace_path(used.realPath) for used in stubs
                                        if 'substitute' not in used.customLayerData),
              'substituted_missing': {workspace_path(used.realPath): used.customLayerData['substitute']
                                      for used in stubs if 'substitute' in used.customLayerData},
              'deactivated_missing': [], 'moved_colliders': [], 'retyped_empty_meshes': 0, 'removed_bodies': 0,
              'disabled_joints': 0, 'removed_articulation_roots': 0, 'uninstanced': [],
              'source_units': units, 'source_up_axis': str(up)}

    def editable(prim):
        """Override of a prim (or path) of the source, or one only the wrapper has (expanded instances)."""
        path = prim if isinstance(prim, Sdf.Path) else prim.GetPath()
        for prefix in path.GetPrefixes()[:-1]:
            if (stage.GetPrimAtPath(prefix) or out.GetPrimAtPath(prefix)).IsInstance() \
                    and str(prefix) not in report['uninstanced']:
                out.OverridePrim(prefix).SetInstanceable(False)
                report['uninstanced'].append(str(prefix))
        return out.OverridePrim(path)

    missing = sorted({str(error.rootSite.path.GetParentPath()) for error in stage.GetCompositionErrors()})
    for path in missing:
        if path in ('/', str(top.GetPath())):
            raise ValueError(f'{source}: {path} itself has unresolved files; vendor them first')
        if not any(path.startswith(f'{done}/') for done in report['deactivated_missing']):
            editable(stage.GetPrimAtPath(path)).SetActive(False)
            report['deactivated_missing'].append(path)
    report['deactivated'] = []
    for path in deactivate:
        if not stage.GetPrimAtPath(path):
            raise ValueError(f'{source}: no prim {path} to deactivate')
        editable(stage.GetPrimAtPath(path)).SetActive(False)
        report['deactivated'].append(path)
    def hand_off_collider(prim, override):
        override.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        override.RemoveAPI(UsdPhysics.CollisionAPI)
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or 'none'
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if child != prim and child.IsA(UsdGeom.Mesh) and _mesh_has_geometry(child) \
                    and not child.HasAPI(UsdPhysics.CollisionAPI):
                target_prim = editable(child)
                UsdPhysics.CollisionAPI.Apply(target_prim)
                UsdPhysics.MeshCollisionAPI.Apply(target_prim).CreateApproximationAttr(approximation)
        report['moved_colliders'].append(str(prim.GetPath()))

    off = report['deactivated_missing'] + report['deactivated']
    gone = tuple(f'{path}/' for path in off)
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if str(prim.GetPath()).startswith(gone) or str(prim.GetPath()) in off:
            continue
        if prim.IsA(UsdGeom.Mesh) and not _mesh_has_geometry(prim):
            # GRScenes groups parts under empty Meshes (some carrying the collider); Newton's mesh import
            # fails on them, so they become Xforms and pass a collider to the meshes below.
            override = editable(prim)
            override.SetTypeName('Xform')
            report['retyped_empty_meshes'] += 1
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                hand_off_collider(prim, override)
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.IsA(UsdGeom.Gprim):
            # Kit PhysX collides the geometry under a collider Xform; UsdPhysics only defines Gprim colliders.
            hand_off_collider(prim, editable(prim))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # Newton ignores physics:rigidBodyEnabled = false and still adds a free body.
            editable(prim).RemoveAPI(UsdPhysics.RigidBodyAPI)
            report['removed_bodies'] += 1
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get() is not False:
            UsdPhysics.Joint(editable(prim)).CreateJointEnabledAttr(False)
            report['disabled_joints'] += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            editable(prim).RemoveAPI(UsdPhysics.ArticulationRootAPI)
            report['removed_articulation_roots'] += 1
    if expand_instancers:
        report['expanded_instancers'] = _expand_instancers(out, top, source, target, bare, editable)
    # Newton hands every shape, visual ones too, to MuJoCo, which rejects a mesh scaled to zero in some
    # axis (hidden parts; deactivated) or mirrored by a negative scale; a mirrored mesh gets its local
    # frame and points both negated, which leaves the surface where it was.
    cache = UsdGeom.XformCache()
    report['degenerate_meshes'], report['unmirrored_meshes'] = [], []
    # Collected first: uninstancing while traversing invalidates the instance proxies being visited.
    found = []
    for prim in Usd.PrimRange(out.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        linear = cache.GetLocalToWorldTransform(prim).ExtractRotationMatrix()
        if min(Gf.Vec3d(*row).GetLength() for row in linear) < 1e-9:
            found.append((prim.GetPath(), None, None, None))
        elif linear.GetDeterminant() < 0:
            mesh = UsdGeom.Mesh(prim)
            found.append((prim.GetPath(), UsdGeom.Xformable(prim).GetLocalTransformation(),
                          mesh.GetPointsAttr().Get(), mesh.GetNormalsAttr().Get()))
    for path, local, points, normals in found:
        if local is None:
            editable(path).SetActive(False)
            report['degenerate_meshes'].append(str(path))
            continue
        flipped = Gf.Matrix4d(local)
        for row in range(3):
            flipped.SetRow(row, -local.GetRow(row))
        mesh = UsdGeom.Mesh(editable(path))
        mesh.ClearXformOpOrder()
        mesh.AddTransformOp().Set(flipped)
        mesh.GetPointsAttr().Set([-point for point in points])
        if normals:
            mesh.GetNormalsAttr().Set([-normal for normal in normals])
        report['unmirrored_meshes'].append(str(path))
    if add_colliders:
        report['added_colliders'] = _add_colliders(out, top, bare, editable)
    report['footprint_boxes'] = {}
    if footprint_colliders:
        import numpy as np
        cache = UsdGeom.XformCache()
        into_top = cache.GetLocalToWorldTransform(top).GetInverse()
        holder = UsdGeom.Scope.Define(out, top.GetPath().AppendChild('footprint_colliders'))
        UsdGeom.Imageable(holder).CreatePurposeAttr(UsdGeom.Tokens.guide)
    for group in footprint_colliders:
        if not (parent := stage.GetPrimAtPath(group)):
            raise ValueError(f'{source}: no prim {group} for footprint colliders')
        for child in parent.GetChildren():
            meshes = [prim for prim in Usd.PrimRange(child, Usd.TraverseInstanceProxies())
                      if prim.IsA(UsdGeom.Mesh) and _mesh_has_geometry(prim)]
            if not meshes or any(prim.HasAPI(UsdPhysics.CollisionAPI)
                                 for prim in Usd.PrimRange(child, Usd.TraverseInstanceProxies())):
                continue
            triangles = []
            for prim in meshes:
                points, faces = _mesh_triangles(prim, cache.GetLocalToWorldTransform(prim) * into_top)
                triangles.append(points[faces])
            boxes, yaw, bottom, top_z = _footprint_boxes(np.concatenate(triangles))
            if not boxes:
                continue
            # One mesh per prim: Newton's MuJoCo conversion grows quadratically with the static shape count.
            corner = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) / 2
            turn = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            points, faces = [], []
            for index, (x, y, size_x, size_y) in enumerate(boxes):
                ring = (corner * (size_x, size_y)) @ turn.T + (x, y)
                points += [(*xy, z) for z in (bottom, top_z) for xy in ring]
                low, high = list(range(8 * index, 8 * index + 4)), list(range(8 * index + 4, 8 * index + 8))
                faces += [low[::-1], high] + [[low[i], low[(i + 1) % 4], high[(i + 1) % 4], high[i]] for i in range(4)]
            mesh = UsdGeom.Mesh.Define(out, holder.GetPath().AppendChild(child.GetName()))
            mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
            mesh.CreateFaceVertexCountsAttr([4] * len(faces))
            mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
            report['footprint_boxes'][str(child.GetPath())] = len(boxes)
    if simplify_colliders:
        report['simplified_colliders'] = _simplify_colliders(out, top, simplify_colliders, editable, units)
    if simplify_visuals or texture_size:
        visuals = Sdf.Layer.CreateNew(str(target.with_suffix('.visuals.usdc')))
        layer.subLayerPaths.append(f'./{Path(visuals.identifier).name}')
        out.SetEditTarget(Usd.EditTarget(visuals))
        if simplify_visuals:
            report['simplified_visuals'] = _simplify_visuals(out, top, int(simplify_visuals), editable)
        if texture_size:
            report['shrunk_textures'] = _shrink_textures(out, top, int(texture_size), editable,
                                                         target.parent / f'{target.stem}.textures')
        out.SetEditTarget(Usd.EditTarget(layer))
        visuals.Save()
    layer.Save()
    check = Usd.Stage.Open(str(target))
    if check.GetCompositionErrors():
        raise ValueError(f'{target}: composition errors remain, e.g. {check.GetCompositionErrors()[0].rootSite}')
    colliders = [prim for prim in Usd.PrimRange(check.GetPseudoRoot(), Usd.TraverseInstanceProxies())
                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        raise ValueError(f'{source}: the scene has no colliders; robots would fall through it')
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                       UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide])
    bounds = Gf.Range3d()
    for prim in colliders:
        bounds.UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
    report.update(colliders=len(colliders),
                  collider_bounds=[[round(v, 3) for v in bounds.GetMin()], [round(v, 3) for v in bounds.GetMax()]])
    return report


def environment(source, output, name=None, substitutes=None, footprint_colliders=(), catalog=None,
                simplify_colliders=None, deactivate=(), simplify_visuals=None, texture_size=None,
                expand_instancers=False, add_colliders=False, bare=BARE_NAMES):
    """Write OUTPUT/usd/NAME.usda wrapping a scene USD as a static environment, and ASSET.json (no MJCF form).
    `substitutes` maps missing file names to replacement USD paths, or is a JSON file of that mapping (paths
    relative to it); `catalog` (shared.environments: key, label, pose, license, options) joins the record."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if substitutes and not isinstance(substitutes, dict):
        mapping = Path(substitutes).resolve()
        substitutes = {name: mapping.parent / path for name, path in json.loads(mapping.read_text()).items()}
    name = name or source.stem
    if (output / 'ASSET.json').exists() or (output / 'usd').exists():
        raise FileExistsError(f'{output} already has ASSET.json or usd/; remove them or choose another output')
    target = output / 'usd' / f'{name}.usda'
    report = environment_layer(source, target, substitutes, footprint_colliders, simplify_colliders, deactivate,
                               simplify_visuals, texture_size, expand_instancers, add_colliders, bare)
    written = [target] + ([target.with_suffix('.visuals.usdc')] if simplify_visuals or texture_size else [])
    record = {
        'source': workspace_path(source), 'source_format': 'usd', 'source_sha256': digest(source),
        'source_layers_sha256': usd_layers(source),
        'formats': {'usd': workspace_path(target)},
        'steps': [{'from': 'usd', 'to': 'usd', 'tool': 'shared.asset_formats.environment_layer', **report}],
        'environment': {'collider_bounds': report['collider_bounds'], **(catalog or {}),
                        'note': 'static environment: place it with scope: global; bounds are in the wrapper frame'},
        'joint_order': [], 'alignment': None,
        'sha256': {path.relative_to(output).as_posix(): digest(path) for path in written},
    }
    (output / 'ASSET.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    return record


def changed_sources(record):
    """Source layers whose content differs from ASSET.json (missing layers included)."""
    current = usd_layers(ROOT / record['source']) if record['source_format'] == 'usd' else {
        record['source']: digest(ROOT / record['source'])}
    recorded = record.get('source_layers_sha256') or {record['source']: record['source_sha256']}
    return sorted(path for path in recorded.keys() | current.keys() if recorded.get(path) != current.get(path))


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
    copy.add_argument('--threads', type=int, default=16, help='URL source: parallel downloads')
    copy.add_argument('--package', action='append', default=[], metavar='NAME=PATH',
                      help='URDF package:// root (default: nearest package.xml with that name)')
    wrap = commands.add_parser('simready', help='Wrap a SimReady USD for Isaac Lab (USD only, no MJCF)')
    wrap.add_argument('source', help='<asset>/simready_usd/sm_<asset>_01.usd of a mirrored SimReady asset, or an asset '
                                     'name (PUBLISHER/ASSET or ASSET, e.g. food_produce_apple_m01): taken from '
                                     'assets/simready/ when present, else downloaded there at the pinned revision')
    wrap.add_argument('output', type=Path, nargs='?',
                      help='Asset directory; writes usd/NAME.usda and ASSET.json (default: assets/<asset>)')
    wrap.add_argument('--name', help='File name of the wrapper (default: SimReady asset name)')
    scene = commands.add_parser('environment', help='Wrap a scene USD as a static environment for Isaac Lab (USD only)')
    scene.add_argument('source', nargs='?',
                       help='Scene USD, or a catalog scene (shared/environments.py: e.g. rivermark, isaac/office, '
                            'grscenes/home/<id>), downloaded when missing; grscenes/<part> downloads that whole part')
    scene.add_argument('output', type=Path, nargs='?', help='Asset directory; writes usd/NAME.usda and ASSET.json')
    scene.add_argument('--list', action='store_true', help='List the catalog scenes')
    scene.add_argument('--name', help='File name of the wrapper (default: source file name)')
    scene.add_argument('--substitutes', type=Path,
                       help='JSON mapping missing upstream file names to replacement USDs (paths relative to it)')
    scene.add_argument('--footprint-colliders', action='append', default=[], metavar='PRIM',
                       help='Give each child of PRIM without colliders (e.g. visual-only buildings) invisible '
                            'boxes over its footprint')
    scene.add_argument('--expand-instancers', action='store_true',
                       help='Turn PointInstancers into referenced copies (Newton ignores them; Unreal exports '
                            'instanced building pieces and foliage as PointInstancers)')
    scene.add_argument('--add-colliders', action='store_true',
                       help='Give visible meshes without a collider a triangle-mesh one (exports without physics, '
                            'e.g. Unreal\'s)')
    scene.add_argument('--bare', default=BARE_NAMES, metavar='REGEX',
                       help='Prim paths that --add-colliders and --expand-instancers leave alone '
                            '(default: vegetation, sky, decals, particles)')
    scene.add_argument('--simplify-colliders', type=float, metavar='CELL',
                       help='Snap collider vertices to a CELL m grid and merge them into a few invisible '
                            'colliders (scenes with millions of collider triangles)')
    scene.add_argument('--simplify-visuals', type=int, metavar='TRIANGLES',
                       help='Cluster the largest visible meshes until the scene shows about TRIANGLES in all '
                            '(Newton views and cameras hold every visible triangle on the GPU)')
    scene.add_argument('--texture-size', type=int, metavar='PIXELS',
                       help='Use copies of the images at most PIXELS on a side (Newton\'s viewer uploads a '
                            'texture per mesh)')
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
            record, copied, unresolved = vendor_url(options.source, options.output, options.base, options.threads)
            origin = record['url']
        else:
            record, copied, unresolved = vendor(options.source, options.output, packages)
            origin = f"{record['repository']}@{record['commit']}"
        print(f"{origin}: {len(record['sha256'])} files; model at {copied}")
        if unresolved:
            print('unresolved (not copied):', ', '.join(map(str, unresolved)))
        return 0
    if options.command == 'environment':
        from shared import environments
        if options.list or options.source is None:
            for item in environments.catalog():
                state = '已包装' if item['wrapped'] else '本地' if item['local'] else '在线'
                print(f"{item['key']:44s} {state:4s} {'' if item['verified'] else '未验证 '}{item['label']}"
                      f"{' · ' + item['note'] if item['note'] else ''}")
            return 0
        if not Path(options.source).is_file():
            key = options.source.strip('/')
            if key.startswith('grscenes/') and key.count('/') == 1:
                scenes = environments.download_grscenes(key.split('/')[1], progress=print)
                print(f'{len(scenes)} 个场景；用 run.py asset environment {key}/<场景 id> 包装')
                return 0
            if options.output or options.name or options.substitutes or options.footprint_colliders \
                    or options.simplify_colliders or options.simplify_visuals or options.texture_size \
                    or options.expand_instancers or options.add_colliders or options.bare != BARE_NAMES:
                parser.error('a catalog scene takes no output, --name, --substitutes, --footprint-colliders, '
                             '--simplify-colliders, --simplify-visuals, --texture-size, --expand-instancers, '
                             '--add-colliders or --bare')
            record = environments.prepare(environments.resolve(key), progress=print)
        else:
            if options.output is None:
                parser.error('a USD source needs an output directory')
            record = environment(options.source, options.output, options.name, options.substitutes,
                                 options.footprint_colliders, simplify_colliders=options.simplify_colliders,
                                 simplify_visuals=options.simplify_visuals, texture_size=options.texture_size,
                                 expand_instancers=options.expand_instancers, add_colliders=options.add_colliders,
                                 bare=options.bare)
        step = record['steps'][0]
        print(json.dumps({**record['formats'], **record['environment']}, indent=2))
        print(f"{step['colliders']} colliders; substituted {len(step['substituted_missing'])} and stubbed "
              f"{len(step['stubbed_missing'])} missing files, deactivated "
              f"{len(step['deactivated_missing'])} prims with unresolved arcs, "
              f"added {sum(step['footprint_boxes'].values())} footprint boxes for "
              f"{len(step['footprint_boxes'])} prims, "
              f"removed {step['removed_bodies']} rigid bodies, disabled {step['disabled_joints']} joints")
        if 'expanded_instancers' in step:
            print(f"expanded {step['expanded_instancers']['instances']} instances of "
                  f"{step['expanded_instancers']['instancers']} PointInstancers")
        if 'added_colliders' in step:
            print(f"added colliders to {step['added_colliders']['meshes']} meshes "
                  f"({step['added_colliders']['triangles']} triangles)")
        return 0
    if options.command == 'simready':
        source = Path(options.source)
        if not source.exists():
            from shared import simready as catalog
            publisher, asset = catalog.resolve(options.source)
            source = catalog.download(publisher, asset, progress=lambda done, total: print(
                f'\r{publisher}/{asset}: {done / 1e6:.1f}/{total / 1e6:.1f} MB', end='', flush=True))
            print()
        output = options.output or ROOT / 'assets' / simready_package(source)['asset']
        record = simready(source, output, options.name)
        step = record['steps'][0]
        print(json.dumps({**record['formats'], **record['simready']}, indent=2))
        if step.get('rigid_body'):
            mass = f"{step['mass']:.4g} kg" if step['mass'] else 'mass as authored'
            print(f"one rigid body at {step['rigid_body']} ({mass}), uninstanced {len(step['uninstanced'])} prims, "
                  f"merged {len(step['merged_bodies'])} bodies, deactivated {len(step['deactivated_joints'])} joints")
        else:
            print(f"uninstanced {len(step['uninstanced'])} prims, retargeted {len(step['retargeted'])} joint ends, "
                  f"articulation root {step['articulation_root'] or 'as authored'}, "
                  f"hard limits on {len(step['hard_limits'])} joints")
        print('joint order:', ', '.join(record['joint_order']) or 'no movable joints')
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
        if 'mjcf' not in record['formats']:
            changed = changed_sources(record) + [
                f'{options.asset}/{name}' for name, hash_ in record['sha256'].items()
                if not (options.asset / name).is_file() or digest(options.asset / name) != hash_]
            if changed:
                print('changed since ASSET.json was written:', *changed, sep='\n  ')
            else:
                print('source layers and generated files match ASSET.json; USD only, no MJCF comparison')
            return 1 if changed or order else 0
        usd, mjcf = ROOT / record['formats']['usd'], ROOT / record['formats']['mjcf']
    else:
        usd, mjcf, order = options.asset, options.mjcf, []
    report = compare(usd, mjcf)
    print(summary(report))
    return 0 if report['aligned'] and not order else 1


if __name__ == '__main__':
    raise SystemExit(main())
