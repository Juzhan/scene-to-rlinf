"""User imports (shared/asset_import.py): uploaded files become USD the scene can use."""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pxr')
trimesh = pytest.importorskip('trimesh')

from shared import asset_import as A  # noqa: E402


def test_mesh_object_is_one_textured_rigid_body_turned_z_up(tmp_path):
    from PIL import Image
    from pxr import Usd, UsdGeom, UsdPhysics
    box = trimesh.creation.box((0.2, 0.4, 0.1))
    image = Image.fromarray((np.arange(48, dtype=np.uint8).reshape(4, 4, 3)))
    box.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(box.vertices), 2)),
                                               material=trimesh.visual.material.PBRMaterial(baseColorTexture=image))
    box.export(tmp_path / 'box.glb')
    report = A.mesh_to_usd([tmp_path / 'box.glb'], tmp_path / 'out' / 'box.usda')
    assert report['size'] == [0.2, 0.1, 0.4] and report['textures'] == 1  # glTF Y-up -> Z-up
    assert report['mass'] == pytest.approx(0.2 * 0.4 * 0.1 * A.DENSITY)
    stage = Usd.Stage.Open(str(tmp_path / 'out' / 'box.usda'))
    root = stage.GetDefaultPrim()
    assert UsdGeom.GetStageUpAxis(stage) == 'Z' and root.HasAPI(UsdPhysics.RigidBodyAPI)
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    render, hull = sorted(meshes, key=lambda prim: 'collisions' in str(prim.GetPath()))
    assert not render.HasAPI(UsdPhysics.CollisionAPI) and hull.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(hull).GetApproximationAttr().Get() == 'convexHull'
    assert (tmp_path / 'out' / 'textures').is_dir()


def test_mesh_scene_is_static_with_triangle_colliders(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    trimesh.creation.box((4000.0, 3000.0, 50.0)).export(tmp_path / 'floor.stl')
    report = A.mesh_to_usd([tmp_path / 'floor.stl'], tmp_path / 'floor.usda', static=True)
    assert report['mass'] is None and report['units_hint'] == 'cm'  # a 4 km scene looks like centimetres
    stage = Usd.Stage.Open(str(tmp_path / 'floor.usda'))
    assert not stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
    mesh = next(prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh))
    assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get() == 'none'


def test_store_keeps_folders_unpacks_zips_and_finds_the_description(tmp_path):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as out:
        out.writestr('arm/urdf/arm.urdf', '<robot name="arm"/>')
        out.writestr('arm/meshes/base.stl', 'solid x\nendsolid x\n')
        out.writestr('__MACOSX/arm/._arm.urdf', 'junk')
    data = archive.getvalue()
    A.store(tmp_path, 'arm.zip', io.BytesIO(data), len(data))
    A.store(tmp_path, '../../escape.obj', io.BytesIO(b'v 0 0 0\n'), 8)  # stays inside source/
    source = tmp_path / 'source'
    assert (source / 'arm/meshes/base.stl').is_file() and not (source / 'arm.zip').exists()
    assert not (source / '__MACOSX').exists() and (source / 'escape.obj').is_file()
    assert A.main_files(source) == ('description', [source / 'arm/urdf/arm.urdf'])
    with pytest.raises(ValueError):
        A.main_files(tmp_path / 'source' / 'arm' / 'missing')


def test_unresolved_references_of_an_import_are_dropped(tmp_path):
    from pxr import Sdf, Usd
    base = Sdf.Layer.CreateNew(str(tmp_path / 'usd' / 'base.usda'))
    stage = Usd.Stage.Open(base)
    stage.DefinePrim('/robot/frame/visuals').GetReferences().AddReference(str(tmp_path / 'usd' / 'base.usda'), '/missing')
    stage.SetDefaultPrim(stage.GetPrimAtPath('/robot'))
    base.Save()
    assert Usd.Stage.Open(str(tmp_path / 'usd' / 'base.usda')).GetCompositionErrors()
    assert A.drop_unresolved(tmp_path / 'usd' / 'base.usda') == ['/robot/frame/visuals']
    assert not Usd.Stage.Open(str(tmp_path / 'usd' / 'base.usda')).GetCompositionErrors()


def test_import_asset_writes_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(A, 'IMPORTED', tmp_path / 'imported')
    directory = A.new_directory('3d crate')
    assert directory.name == 'asset_3d_crate' and A.new_directory('3d crate').name == 'asset_3d_crate_2'
    mesh = trimesh.creation.box((0.1, 0.1, 0.1))
    data = mesh.export(file_type='obj').encode()
    A.store(directory, 'crate.obj', io.BytesIO(data), len(data))
    record = A.import_asset(directory, 'object', progress=lambda message: None)
    assert record['usd'].endswith('usd/asset_3d_crate.usda') and Path(record['usd']).is_file()
    assert json.loads((directory / 'IMPORT.json').read_text())['kind'] == 'object'


def test_a_folder_of_description_variants_imports_the_complete_one(tmp_path):
    source = tmp_path / 'source'
    (source / 'meshes').mkdir(parents=True)
    (source / 'meshes' / 'base.obj').write_text('v 0 0 0\n')
    (source / 'robot.urdf').write_text('<robot><link><collision><geometry><mesh filename="meshes/collision/base.obj"/>'
                                       '</geometry></collision></link></robot>')
    (source / 'robot_original.urdf').write_text('<robot><link><visual><geometry><mesh filename="meshes/base.obj"/>'
                                                '</geometry></visual></link></robot>')
    assert A.missing_files(source / 'robot.urdf') == 1 and A.missing_files(source / 'robot_original.urdf') == 0
    assert A.main_files(source) == ('description', [source / 'robot_original.urdf'])


def test_a_urdf_out_of_its_ros_workspace_finds_its_meshes_and_says_what_is_missing(tmp_path):
    """package:// without package.xml (two packages flattened into one folder), a collision mesh not uploaded."""
    from xml.etree import ElementTree
    source = tmp_path / 'source'
    (source / 'meshes' / 'arm' / 'visual').mkdir(parents=True)
    (source / 'meshes' / 'arm' / 'visual' / 'link1.stl').write_text('solid\n')
    (source / 'meshes' / 'finger.STL').write_text('solid\n')
    (source / 'arm.urdf').write_text(
        '<robot name="arm"><link name="link1">'
        '<visual><origin xyz="0 0 .1"/><geometry><mesh filename="package://arm_description/meshes/arm/visual/link1.stl"/></geometry></visual>'
        '<collision><geometry><mesh filename="package://arm_description/meshes/arm/collision/link1.obj"/></geometry></collision></link>'
        '<link name="finger"><visual><geometry><mesh filename="package://gripper/meshes/finger.STL"/></geometry></visual></link>'
        '<joint name="drive" type="revolute"><parent link="link1"/><child link="finger"/></joint>'
        '<joint name="follow" type="revolute"><parent link="link1"/><child link="finger"/><mimic joint="drive" multiplier="-1"/></joint>'
        '</robot>')
    report = A.prepare_urdf(source / 'arm.urdf', source, tmp_path / 'urdf' / 'arm.urdf')
    assert report['meshes_found'] == 2 and report['collision_from_visual'] == ['link1']
    assert report['missing_meshes'] == ['package://arm_description/meshes/arm/collision/link1.obj']
    robot = ElementTree.parse(tmp_path / 'urdf' / 'arm.urdf').getroot()
    collision = robot.find('link/collision')
    assert collision.find('geometry/mesh').get('filename') == '../source/meshes/arm/visual/link1.stl'
    assert collision.find('origin').get('xyz') == '0 0 .1'  # with the visual's placement
    assert A.mimic_joints(source / 'arm.urdf') == {'follow': {'leader': 'drive', 'ratio': -1.0, 'offset': 0.0}}
    # A description that converts as it is gets no copy.
    (source / 'plain.urdf').write_text('<robot name="p"><link name="a"><visual><geometry><mesh filename="meshes/finger.STL"/>'
                                       '</geometry></visual></link></robot>')
    assert A.prepare_urdf(source / 'plain.urdf', source, tmp_path / 'urdf' / 'plain.urdf') == {}
    # Moments no body can have (1 + 2 < 5) are balanced as the MuJoCo compiler does; possible ones stay.
    robot = ElementTree.fromstring('<robot><link name="bad"><inertial><inertia ixx="1" iyy="2" izz="5" ixy="0" ixz="0" iyz="0"/>'
                                   '</inertial></link><link name="good"><inertial><inertia ixx="1" iyy="2" izz="3" ixy="0" ixz="0" iyz="0"/>'
                                   '</inertial></link></robot>')
    assert A.balance_inertia(robot) == ['bad']
    assert robot.find('link/inertial/inertia').get('izz') == '2.66666667' and robot.find('link/inertial/inertia').get('ixx') == '2.66666667'


def test_a_urdf_without_its_meshes_stops_before_converting(tmp_path):
    (tmp_path / 'source').mkdir()
    (tmp_path / 'source' / 'arm.urdf').write_text('<robot name="arm"><link name="base"><visual><geometry>'
                                                  '<mesh filename="link_base.STL"/></geometry></visual></link></robot>')
    with pytest.raises(A.ImportFailed, match='整个文件夹') as failed:
        A.prepare_urdf(tmp_path / 'source' / 'arm.urdf', tmp_path / 'source', tmp_path / 'urdf' / 'arm.urdf')
    assert 'link_base.STL' in failed.value.detail


def test_a_ros_xacro_is_expanded_with_its_packages_found_in_the_upload(tmp_path):
    import os
    import subprocess
    from shared.assembly.runtimes import declared_python
    python = str(declared_python('isaac', A.ROOT))
    if subprocess.run([python, '-c', 'import xacro'], capture_output=True).returncode:
        pytest.skip('xacro is not installed in the Isaac interpreter')
    package = tmp_path / 'source' / 'arm_description'
    (package / 'urdf').mkdir(parents=True)
    (package / 'meshes').mkdir()
    (package / 'meshes' / 'base.stl').write_text('solid\n')
    (package / 'package.xml').write_text('<package><name>arm_description</name></package>')
    (package / 'urdf' / 'parts.xacro').write_text(
        '<robot xmlns:xacro="http://ros.org/wiki/xacro"><xacro:macro name="base" params="size">'
        '<link name="base_${size}"><visual><geometry><mesh filename="package://arm_description/meshes/base.stl"/></geometry></visual></link>'
        '</xacro:macro></robot>')
    (package / 'urdf' / 'arm.urdf.xacro').write_text(
        '<robot xmlns:xacro="http://ros.org/wiki/xacro" name="arm"><xacro:arg name="size" default="small"/>'
        '<xacro:include filename="$(find arm_description)/urdf/parts.xacro"/><xacro:base size="$(arg size)"/></robot>')
    assert A.main_files(tmp_path / 'source') == ('description', [package / 'urdf' / 'arm.urdf.xacro'])
    given, packages = A.expand_xacro(package / 'urdf' / 'arm.urdf.xacro', tmp_path / 'source', tmp_path / 'urdf' / 'arm.urdf', 'size:=large')
    assert given == {'size': 'large'} and packages == {'arm_description': 'arm_description'}
    text = (tmp_path / 'urdf' / 'arm.urdf').read_text()
    assert 'base_large' in text and 'package://arm_description/meshes/base.stl' in text
    report = A.prepare_urdf(tmp_path / 'urdf' / 'arm.urdf', tmp_path / 'source', tmp_path / 'urdf' / 'arm.urdf', package / 'urdf')
    assert report['meshes_found'] == 1
    assert '../source/arm_description/meshes/base.stl' in (tmp_path / 'urdf' / 'arm.urdf').read_text()
    # A xacro that only defines macros has no robot of its own: said before the importer sees an empty <robot/>.
    with pytest.raises(A.ImportFailed, match='只定义了宏'):
        A.needed_files(tmp_path / 'source', 'arm_description/urdf/parts.xacro')
    # A package the upload does not hold is named in the message.
    (package / 'urdf' / 'other.urdf.xacro').write_text(
        '<robot xmlns:xacro="http://ros.org/wiki/xacro" name="o"><xacro:include filename="$(find gripper_description)/g.xacro"/></robot>')
    with pytest.raises(A.ImportFailed, match='gripper_description'):
        A.expand_xacro(package / 'urdf' / 'other.urdf.xacro', tmp_path / 'source', tmp_path / 'urdf' / 'other.urdf')


def test_from_a_chosen_folder_only_the_files_a_model_uses_are_needed(tmp_path):
    """Other files only exist (empty stand-ins); what a URDF, an MJCF and an OBJ name is what gets sent."""
    root = tmp_path / 'repo'
    for name in ('arm_description/meshes/base.stl', 'arm_description/meshes/unused.stl', 'other_robot/meshes/big.stl',
                 'menagerie/ur5e/assets/base.obj', 'menagerie/ur5e/assets/unused.obj', 'menagerie/ur5e/textures/wood.png',
                 'props/cup.obj', 'props/cup.png', 'props/plate.obj'):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).touch()
    (root / 'arm_description' / 'package.xml').write_text('<package><name>arm_description</name></package>')
    (root / 'arm_description' / 'urdf').mkdir()
    (root / 'arm_description' / 'urdf' / 'arm.urdf').write_text(
        '<robot name="arm"><link name="base"><visual><geometry><mesh filename="package://arm_description/meshes/base.stl"/>'
        '</geometry></visual></link></robot>')
    assert A.needed_files(root, 'arm_description/urdf/arm.urdf') == ['arm_description/meshes/base.stl', 'arm_description/urdf/arm.urdf']
    # Menagerie: scene.xml includes the robot file, whose <compiler> sets the mesh folder.
    (root / 'menagerie' / 'ur5e' / 'scene.xml').write_text('<mujoco><include file="ur5e.xml"/></mujoco>')
    (root / 'menagerie' / 'ur5e' / 'ur5e.xml').write_text(
        '<mujoco><compiler meshdir="assets" texturedir="textures"/><asset><mesh file="base.obj"/>'
        '<texture name="w" file="wood.png"/></asset></mujoco>')
    assert A.needed_files(root, 'menagerie/ur5e/scene.xml') == [
        'menagerie/ur5e/assets/base.obj', 'menagerie/ur5e/scene.xml', 'menagerie/ur5e/textures/wood.png', 'menagerie/ur5e/ur5e.xml']
    (root / 'props' / 'cup.obj').write_text('mtllib cup.mtl\nv 0 0 0\n')
    (root / 'props' / 'cup.mtl').write_text('newmtl a\nmap_Kd cup.png\n')
    assert A.needed_files(root, 'props/cup.obj') == ['props/cup.mtl', 'props/cup.obj', 'props/cup.png']
    # A mesh the URDF names that is not in the folder stops it here, before anything large is sent.
    (root / 'arm_description' / 'urdf' / 'broken.urdf').write_text(
        '<robot name="b"><link name="base"><visual><geometry><mesh filename="meshes/gone.stl"/></geometry></visual></link></robot>')
    with pytest.raises(A.ImportFailed, match='整个文件夹'):
        A.needed_files(root, 'arm_description/urdf/broken.urdf')


def test_massless_virtual_links_get_a_tiny_mass_in_a_wrapper(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(tmp_path / 'robot.usda'))
    root = UsdGeom.Xform.Define(stage, '/robot').GetPrim()
    stage.SetDefaultPrim(root)
    virtual = UsdGeom.Xform.Define(stage, '/robot/base_x').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(virtual)
    body = UsdGeom.Xform.Define(stage, '/robot/base_link').GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, '/robot/base_link/box').GetPrim())
    stage.GetRootLayer().Save()
    assert A.wrap_for_isaac(tmp_path / 'robot.usda', tmp_path / 'wrapped' / 'robot.usda') == {'massless_bodies': ['/robot/base_x']}
    wrapped = Usd.Stage.Open(str(tmp_path / 'wrapped' / 'robot.usda'))
    assert UsdPhysics.MassAPI(wrapped.GetPrimAtPath('/robot/base_x')).GetMassAttr().Get() == pytest.approx(A.TINY_MASS)
    assert not wrapped.GetPrimAtPath('/robot/base_link').HasAPI(UsdPhysics.MassAPI)  # its colliders give it mass
    assert A.wrap_for_isaac(tmp_path / 'wrapped' / 'robot.usda', tmp_path / 'again.usda') == {}


def test_mjcf_import_leftovers_are_fixed_in_the_wrapper(tmp_path):
    """The Isaac MJCF importer writes a hinge's damping (LIBERO's microwave door) as a drive with maxForce 0:
    PhysX applies none of it and Newton's MuJoCo solver rejects the empty force range. Colliders inside
    instances still give their bodies mass."""
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(tmp_path / 'oven.usda'))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/oven').GetPrim())
    box = UsdGeom.Cube.Define(stage, '/proto/box').GetPrim()
    UsdPhysics.CollisionAPI.Apply(box)
    for name in ('body', 'door'):
        prim = UsdGeom.Xform.Define(stage, f'/oven/{name}').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        shape = UsdGeom.Xform.Define(stage, f'/oven/{name}/collisions').GetPrim()
        shape.GetReferences().AddInternalReference('/proto')
        shape.SetInstanceable(True)
    hinge = UsdPhysics.RevoluteJoint.Define(stage, '/oven/hinge')
    hinge.CreateBody0Rel().SetTargets(['/oven/body'])
    hinge.CreateBody1Rel().SetTargets(['/oven/door'])
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), 'angular')
    drive.CreateDampingAttr(0.5)
    drive.CreateMaxForceAttr(0.0)
    UsdGeom.Xform.Define(stage, '/oven/body/sites')
    UsdGeom.Cube.Define(stage, '/oven/body/sites/heating_region')  # an MJCF site: a marker, not a part
    stage.GetRootLayer().Save()
    report = A.wrap_for_isaac(tmp_path / 'oven.usda', tmp_path / 'wrapped.usda')
    assert report == {'unlimited_drives': ['/oven/hinge:angular'], 'hidden_sites': ['/oven/body/sites']}, report
    wrapped = Usd.Stage.Open(str(tmp_path / 'wrapped.usda'))
    drive = UsdPhysics.DriveAPI(wrapped.GetPrimAtPath('/oven/hinge'), 'angular')
    assert drive.GetMaxForceAttr().Get() == pytest.approx(A.FLT_MAX) and drive.GetDampingAttr().Get() == 0.5
    site = UsdGeom.Imageable(wrapped.GetPrimAtPath('/oven/body/sites/heating_region'))
    assert site.ComputeVisibility() == UsdGeom.Tokens.invisible


def test_a_turned_root_body_takes_the_asset_frame_keeping_world_poses(tmp_path):
    """What the Isaac MJCF importer makes of MuJoCo Menagerie's UR5e: a base yawed 180 degrees, welded to the
    world, and an empty worldBody articulation root beside the base's."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    from shared.asset_formats import usd_root_offset
    stage = Usd.Stage.CreateNew(str(tmp_path / 'arm.usda'))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/arm').GetPrim())
    base = UsdGeom.Xform.Define(stage, '/arm/base')
    base.AddOrientOp().Set(Gf.Quatf(0, 0, 0, 1))
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    link = UsdGeom.Xform.Define(stage, '/arm/base/link')
    link.AddTranslateOp().Set(Gf.Vec3d(0.3, 0, 0.1))
    UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
    for parent in (base, link):
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage, parent.GetPath().AppendChild('box')).GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/arm/base/link/hinge')
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([link.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.3, 0, 0.1))
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, '/arm/worldBody').GetPrim())
    UsdPhysics.FixedJoint.Define(stage, '/arm/root_joint').CreateBody1Rel().SetTargets([base.GetPath()])
    stage.GetRootLayer().Save()
    assert usd_root_offset(tmp_path / 'arm.usda') == pytest.approx([0, 0, 0])  # turned, not moved
    report = A.wrap_for_isaac(tmp_path / 'arm.usda', tmp_path / 'wrapped.usda')
    assert report['rerooted']['body'] == '/arm/base' and report['stray_roots'] == ['/arm/worldBody']
    assert report['world_joints'] == ['/arm/root_joint']
    assert usd_root_offset(tmp_path / 'wrapped.usda') is None
    assert not Usd.Stage.Open(str(tmp_path / 'wrapped.usda')).GetPrimAtPath('/arm/worldBody').HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not Usd.Stage.Open(str(tmp_path / 'wrapped.usda')).GetPrimAtPath('/arm/root_joint').IsActive()
    world = lambda path, prim: UsdGeom.XformCache().GetLocalToWorldTransform(Usd.Stage.Open(str(path)).GetPrimAtPath(prim))
    assert Gf.IsClose(world(tmp_path / 'wrapped.usda', '/arm/base/link'), world(tmp_path / 'arm.usda', '/arm/base/link'), 1e-6)
    wrapped = Usd.Stage.Open(str(tmp_path / 'wrapped.usda'))
    frame = UsdPhysics.Joint(wrapped.GetPrimAtPath('/arm/base/link/hinge'))
    assert frame.GetLocalPos0Attr().Get() == pytest.approx(Gf.Vec3f(-0.3, 0, 0.1), abs=1e-6)


def test_joints_without_drive_stiffness_are_reported(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(tmp_path / 'arm.usda'))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/arm').GetPrim())
    for name, stiffness in (('shoulder', 0.0), ('elbow', 50.0), ('wrist', None)):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f'/arm/{name}').GetPrim()
        if stiffness is not None:
            UsdPhysics.DriveAPI.Apply(joint, 'angular').CreateStiffnessAttr(stiffness)
    stage.GetRootLayer().Save()
    assert A.undriven_joints(tmp_path / 'arm.usda') == ['shoulder', 'wrist']


def test_an_open_mesh_collides_through_its_hull_with_authored_inertia(tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    box = trimesh.creation.box((0.4, 0.2, 0.1))
    box.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [0, 0, 1]))
    box.apply_translation((0.1, 0.05, 0.0))
    open_box = trimesh.Trimesh(box.vertices, box.faces[2:], process=False)  # two faces missing: not watertight
    open_box.export(tmp_path / 'open.ply')
    report = A.mesh_to_usd([tmp_path / 'open.ply'], tmp_path / 'open.usda', mass=2.0)
    assert report['mass'] == 2.0
    stage = Usd.Stage.Open(str(tmp_path / 'open.usda'))
    render = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and 'collisions' not in str(p.GetPath()))
    collider = next(p for p in stage.Traverse() if 'collisions' in str(p.GetPath()) and p.IsA(UsdGeom.Mesh))
    assert not render.HasAPI(UsdPhysics.CollisionAPI) and collider.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdGeom.Imageable(collider).ComputePurpose() == UsdGeom.Tokens.guide
    properties = UsdPhysics.MassAPI(stage.GetDefaultPrim())
    hull = box.convex_hull
    assert np.allclose(properties.GetCenterOfMassAttr().Get(), hull.center_mass, atol=1e-5)
    rotation = np.array(Gf.Matrix3d(Gf.Rotation(properties.GetPrincipalAxesAttr().Get())))  # rows: principal axes
    tensor = rotation.T @ np.diag(properties.GetDiagonalInertiaAttr().Get()) @ rotation
    assert np.allclose(tensor, hull.moment_inertia * 2.0 / hull.volume, atol=1e-6)

pytestmark = pytest.mark.group_import  # runs in the [import] dependency group
