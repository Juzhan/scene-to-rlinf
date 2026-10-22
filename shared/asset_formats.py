"""Keep one robot/object asset available to both Isaac (USD) and MuJoCo/mjlab (MJCF).

Every conversion is done by an official converter; this module chains them through
URDF, stages meshes, and records provenance plus an alignment report in ASSET.json:

    USD  -> URDF : NVIDIA UsdToUrdf (Isaac Sim isaacsim.asset.exporter.urdf; pxr only)
    URDF -> MJCF : MuJoCo URDF compiler
    URDF -> USD  : Isaac Lab UrdfConverter (Isaac Sim URDF importer; starts Kit)
    MJCF -> USD  : Isaac Lab MjcfConverter (Isaac Sim MJCF importer; starts Kit)

The body must align (joints, limits, axes, masses, inertia, frames); appearance only
needs to be close. Links with collision geometry are drawn in MJCF from that geometry,
coloured with the dominant visual material colour. Actuators and gains are not part of
the converted files: each backend's asset factory declares them.

    run.py asset convert assets/piper_x/piper_x.usda assets/piper_x
    run.py asset check assets/piper_x

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
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

ROOT = Path(__file__).resolve().parents[1]
USD_SUFFIXES = {'.usd', '.usda', '.usdc'}
MESH_SUFFIXES = {'.stl'}
TOLERANCE = {'position': 1e-4, 'mass': 1e-3, 'inertia': 1e-3, 'angle': 1e-4, 'axis': 1e-4}


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


def usd_to_urdf(source, directory, name, root=None):
    """Export the USD kinematic tree; link names lose the exporter's '<defaultPrim>_' prefix."""
    from pxr import Usd
    UsdToUrdf, extension = _usd_to_urdf_library()
    stage = Usd.Stage.Open(str(source))
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
                  'renamed_links': rename}


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
    colours, dropped, written, by_content = {}, [], {}, {}
    for link in robot.iter('link'):
        visuals = link.findall('visual')
        if visuals and link.find('collision/geometry') is not None and not keep_visuals:
            # MJCF is not the rendering format: draw the link with its collision
            # geometry in the dominant visual colour.
            colour = next(filter(None, (urdf_colour(visual, source, packages) for visual in visuals)), None)
            if colour is not None:
                colours[link.get('name')] = colour
            for visual in visuals:
                link.remove(visual)
                dropped.append(link.get('name'))
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
                if path.suffix.lower() in MESH_SUFFIXES:
                    shutil.copyfile(path, target)
                else:
                    import trimesh
                    # Geometry only: MuJoCo meshes carry no DAE/OBJ material or texture.
                    trimesh.load(path, force='mesh').export(target)
                written[key] = by_content[content] = target.relative_to(directory).as_posix()
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
    for body, colour in colours.items():
        for geom in spec.body(body).geoms:
            if geom.contype or geom.conaffinity:
                geom.rgba = colour
    spec.compile()
    (directory / (name + '.xml')).write_text(spec.to_xml())
    return directory / (name + '.xml'), {
        'tool': 'MuJoCo URDF compiler', 'version': mujoco.__version__,
        'packages': {key: workspace_path(value) for key, value in packages.items()},
        'visuals_replaced_by_collision_geometry': sorted(set(dropped)),
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
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(usd))
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
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(usd))
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
    """Rest-pose rigid bodies and joints read directly with pxr."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    bodies, joints = {}, {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            transform = cache.GetLocalToWorldTransform(prim)
            mass = UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI) else None
            rotation = transform.RemoveScaleShear().ExtractRotationQuat()
            entry = {'position': list(transform.ExtractTranslation()),
                     'orientation': [rotation.GetReal(), *rotation.GetImaginary()]}
            if mass is not None and mass.GetMassAttr().HasAuthoredValue():
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
                joints[prim.GetName()] = {
                    'type': kind, 'axis': list(Gf.Rotation(rotation).TransformDir(Gf.Vec3d(*axis))),
                    'range': [joint.GetLowerLimitAttr().Get() * scale, joint.GetUpperLimitAttr().Get() * scale]}
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
    kinds = {mujoco.mjtJoint.mjJNT_HINGE: 'hinge', mujoco.mjtJoint.mjJNT_SLIDE: 'slide'}
    joints = {model.joint(index).name: {
        'type': kinds.get(model.jnt_type[index], 'other'), 'axis': data.xaxis[index].tolist(),
        'range': model.jnt_range[index].tolist() if model.jnt_limited[index] else None}
        for index in range(model.njnt)}
    return bodies, joints


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
    for name in sorted(usd_joint.keys() | mj_joint.keys()):
        if name not in usd_joint or name not in mj_joint:
            mismatches.append({'item': f'joint {name}', 'kind': 'missing',
                               'usd': name in usd_joint, 'mjcf': name in mj_joint})
            continue
        a, b = usd_joint[name], mj_joint[name]
        if a['type'] != b['type'] or b['range'] is None:
            mismatches.append({'item': f'joint {name}', 'kind': 'type/limit', 'usd': a, 'mjcf': b})
            continue
        check('axis', f'joint {name}', a['axis'], b['axis'])
        check('angle', f'joint {name} range', a['range'], b['range'])
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
        'not_compared': ['collision shapes (MuJoCo uses mesh convex hulls; PhysX follows the USD collider '
                         'approximation)', 'contact/friction parameters', 'actuators (declared by backend factories)'],
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


def recorded_mjcf(source):
    """MJCF recorded by `asset convert` next to a USD/URDF source, or None.

    A record whose source changed afterwards is an error, never silently reused."""
    source = Path(source).resolve()
    for record_path in (source.parent / 'ASSET.json', source.parent.parent / 'ASSET.json'):
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
    for key in ('usd_only_bodies', 'mjcf_only_bodies'):
        if report[key]:
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
                        help='Keep visual meshes in MJCF even where collision geometry exists')
    verify = commands.add_parser('check', help='Compare the USD and MJCF forms of an asset')
    verify.add_argument('asset', type=Path, help='Asset directory with ASSET.json, or a USD file')
    verify.add_argument('mjcf', type=Path, nargs='?', help='MJCF file when the first argument is a USD file')
    options = parser.parse_args(argv)
    if options.command == 'convert':
        packages = {}
        for item in options.package:
            key, separator, value = item.partition('=')
            if not separator:
                parser.error('--package expects NAME=PATH')
            packages[key] = value
        record = convert(options.source, options.output, options.name, packages, options.root,
                         options.fix_base, options.keep_visuals)
        print(json.dumps(record['formats'], indent=2))
        print(summary(record['alignment']))
        return 0 if record['alignment']['aligned'] else 1
    if options.mjcf is None:
        formats = json.loads((options.asset / 'ASSET.json').read_text())['formats']
        usd, mjcf = ROOT / formats['usd'], ROOT / formats['mjcf']
    else:
        usd, mjcf = options.asset, options.mjcf
    report = compare(usd, mjcf)
    print(summary(report))
    return 0 if report['aligned'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
