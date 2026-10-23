import json
from pathlib import Path

import pytest

pytest.importorskip('mujoco')
pytest.importorskip('pxr')

from shared.asset_formats import (compare, physx_stage, urdf_to_mjcf, usd_capsules_into_mjcf,
                                  usd_frames_into_mjcf)

INERTIAL = '<inertial><origin xyz="{com}"/><mass value="{mass}"/><inertia ixx="0.01" iyy="0.02" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>'


def write_urdf(directory, com='0 0 0.1', mass=1.0, rpy='1.5707963 0 0', visual_rpy='0 0 0', joint='revolute'):
    (directory / 'part.obj').write_text(
        'mtllib part.mtl\nv 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n'
        'usemtl red\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n')
    (directory / 'part.mtl').write_text('newmtl red\nKd 0.9 0.1 0.1\n')
    path = directory / 'robot.urdf'
    path.write_text(f'''<robot name="robot">
  <link name="base">{INERTIAL.format(com='0 0 0', mass=2.0)}
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="arm">{INERTIAL.format(com=com, mass=mass)}
    <visual><origin rpy="{visual_rpy}"/><geometry><mesh filename="part.obj"/></geometry></visual>
    <collision><geometry><mesh filename="part.obj"/></geometry></collision></link>
  <joint name="hinge" type="{joint}"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.2" rpy="{rpy}"/><axis xyz="0 0 1"/>
    <limit {'' if joint == 'continuous' else 'lower="-1" upper="1" '}effort="1" velocity="1"/></joint>
</robot>''')
    return path


def write_usd(path, com=(0.0, 0.0, 0.1), mass=1.0, limited=True):
    """The write_urdf robot: a box collider on base, a mesh collider on arm."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, '/robot')
    for name, translate, rotate, body_mass, centre in (
            ('base', (0, 0, 0), Gf.Quatf(1), 2.0, (0, 0, 0)),
            ('arm', (0, 0, 0.2), Gf.Quatf(0.7071068, 0.7071068, 0, 0), mass, com)):
        xform = UsdGeom.Xform.Define(stage, f'/robot/{name}')
        xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
        xform.AddOrientOp().Set(rotate)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
        api.CreateMassAttr(body_mass)
        api.CreateCenterOfMassAttr(Gf.Vec3f(*centre))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.02, 0.03))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1))
        if name == 'base':
            UsdGeom.Cube.Define(stage, '/robot/base/collider').CreateSizeAttr(0.1)
        else:
            mesh = UsdGeom.Mesh.Define(stage, '/robot/arm/collider')  # part.obj
            mesh.CreatePointsAttr([(0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)])
            mesh.CreateFaceVertexCountsAttr([3] * 4)
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3])
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f'/robot/{name}/collider'))
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/robot/hinge')
    joint.CreateBody0Rel().SetTargets(['/robot/base'])
    joint.CreateBody1Rel().SetTargets(['/robot/arm'])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))
    joint.CreateLocalRot0Attr(Gf.Quatf(0.7071068, 0.7071068, 0, 0))
    joint.CreateAxisAttr('Z')
    if limited:
        joint.CreateLowerLimitAttr(-57.29578)
        joint.CreateUpperLimitAttr(57.29578)
    stage.Save()
    return path


def test_urdf_to_mjcf_keeps_body_and_colours_collision_geometry(tmp_path):
    import mujoco
    mjcf, step = urdf_to_mjcf(write_urdf(tmp_path), tmp_path / 'mjcf', 'robot')
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    arm = model.body('arm').id
    geoms = [index for index in range(model.ngeom) if model.geom_bodyid[index] == arm]
    assert step['visuals_replaced_by_collision_geometry'] == ['arm']
    assert step['visuals_kept'] == {} and step['meshes_simplified'] == {}
    assert len(geoms) == 1 and model.geom_rgba[geoms[0]] == pytest.approx([0.9, 0.1, 0.1, 1.0])
    assert sorted(path.name for path in (tmp_path / 'mjcf/assets').iterdir()) == ['part.stl']
    report = compare(write_usd(tmp_path / 'robot.usda'), mjcf)
    assert report['aligned'], report['mismatches']


def test_urdf_to_mjcf_keeps_visual_that_collision_geometry_misplaces(tmp_path):
    import mujoco
    mjcf, step = urdf_to_mjcf(write_urdf(tmp_path, visual_rpy='0 0 1.5707963'), tmp_path / 'mjcf', 'robot')
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    geoms = [index for index in range(model.ngeom) if model.geom_bodyid[index] == model.body('arm').id]
    assert step['visuals_replaced_by_collision_geometry'] == [] and step['visuals_kept'] == {'arm': 0.1}
    visual = [index for index in geoms if not model.geom_contype[index]]
    collision = [index for index in geoms if model.geom_contype[index]]
    assert len(visual) == len(collision) == 1
    assert model.geom_group[visual[0]] < 3
    assert model.geom_rgba[visual[0]] == pytest.approx([0.9, 0.1, 0.1, 1.0])
    assert model.geom_group[collision[0]] == 3


def test_meshes_over_the_face_limit_are_reduced(tmp_path, monkeypatch):
    import trimesh
    import shared.asset_formats as formats
    monkeypatch.setattr(formats, 'MESH_FACE_LIMIT', 500)
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.05)
    sphere.export(tmp_path / 'ball.stl')
    reduced = formats.write_mesh(tmp_path / 'ball.stl', tmp_path / 'out.stl')
    mesh = trimesh.load(tmp_path / 'out.stl', force='mesh')
    assert reduced[0] == len(sphere.faces) and reduced[1] == len(mesh.faces) <= 500
    assert mesh.bounds == pytest.approx(sphere.bounds, abs=2 * reduced[2])


def test_compare_reports_misplaced_centre_of_mass(tmp_path):
    mjcf, _ = urdf_to_mjcf(write_urdf(tmp_path, com='0 0.01 0.1'), tmp_path / 'mjcf', 'robot')
    report = compare(write_usd(tmp_path / 'robot.usda'), mjcf)
    assert not report['aligned']
    assert [item['item'] for item in report['mismatches']] == ['body arm centre of mass']


def test_turned_body_frame_is_restored_without_moving_geometry(tmp_path):
    import mujoco
    # The arm link frame is turned by pi about its z axis; COM, inertia and axis are symmetric to it.
    mjcf, _ = urdf_to_mjcf(write_urdf(tmp_path, rpy='-1.5707963 0 3.1415927'), tmp_path / 'mjcf', 'robot')
    usd = write_usd(tmp_path / 'robot.usda')
    report = compare(usd, mjcf)
    assert [item['item'] for item in report['mismatches']] == ['body arm frame orientation']

    def geometry(path):
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        data.qpos[:] = 0.3
        mujoco.mj_kinematics(model, data)
        return data.geom_xpos.copy(), data.geom_xmat.copy()

    before = geometry(mjcf)
    assert usd_frames_into_mjcf(mjcf, usd) == ['arm']
    assert compare(usd, mjcf)['aligned']
    for old, new in zip(before, geometry(mjcf)):
        assert new == pytest.approx(old, abs=1e-6)


def test_continuous_joints_align_without_limits(tmp_path):
    mjcf, _ = urdf_to_mjcf(write_urdf(tmp_path, joint='continuous'), tmp_path / 'mjcf', 'robot')
    assert compare(write_usd(tmp_path / 'robot.usda', limited=False), mjcf)['aligned']
    report = compare(write_usd(tmp_path / 'limited.usda'), mjcf)
    assert [item['kind'] for item in report['mismatches']] == ['type/limit']


def test_capsule_colliders_dropped_by_urdf_come_from_usd(tmp_path):
    import mujoco
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    mjcf, _ = urdf_to_mjcf(write_urdf(tmp_path, visual_rpy='0 0 1.5707963'), tmp_path / 'mjcf', 'robot')
    usd = write_usd(tmp_path / 'robot.usda')
    stage = Usd.Stage.Open(str(usd))
    capsule = UsdGeom.Capsule.Define(stage, '/robot/arm/wheel')
    capsule.AddTranslateOp().Set(Gf.Vec3d(0.01, 0.02, 0.05))
    capsule.AddScaleOp().Set(Gf.Vec3f(2, 1, 1))
    capsule.CreateAxisAttr('X')
    capsule.CreateRadiusAttr(0.03)
    capsule.CreateHeightAttr(0.1)
    UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
    stage.Save()
    report = compare(usd, mjcf)
    assert [(item['item'], item['usd'], item['mjcf']) for item in report['mismatches']] == [
        ('body arm colliders', {'capsule': 1, 'mesh': 1}, {'mesh': 1})]

    assert usd_capsules_into_mjcf(mjcf, usd) == {'arm': 1}
    assert compare(usd, mjcf)['aligned']
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    index = next(i for i in range(model.ngeom) if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE)
    world = UsdGeom.XformCache().GetLocalToWorldTransform(capsule.GetPrim())
    assert data.geom_xpos[index] == pytest.approx(list(world.ExtractTranslation()), abs=1e-6)
    axis = world.RemoveScaleShear().TransformDir(Gf.Vec3d(1, 0, 0))
    assert data.geom_xmat[index].reshape(3, 3)[:, 2] == pytest.approx(list(axis), abs=1e-6)
    assert model.geom_size[index][:2] == pytest.approx([0.03, 0.1])  # radius; half length scaled by 2
    assert model.geom_group[index] == 3  # hidden like the arm's other collider: its visual mesh is kept


def test_fixed_joint_child_takes_the_pose_physx_simulates(tmp_path):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    path = tmp_path / 'feet.usda'
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, '/robot')
    for name, z in (('shank', 0.), ('foot', -.3), ('loose', -.6), ('hinged', -.9)):
        body = UsdGeom.Xform.Define(stage, f'/robot/{name}')
        body.AddTranslateOp().Set(Gf.Vec3d(0, 0, z))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    turns = {'foot': .0012, 'loose': .1, 'hinged': .0012}  # rad about x; authored on the child only
    for child, turn in turns.items():
        UsdGeom.Xformable(stage.GetPrimAtPath(f'/robot/{child}')).AddRotateXOp().Set(turn * 180 / 3.141592653589793)
        joint = (UsdPhysics.RevoluteJoint if child == 'hinged' else UsdPhysics.FixedJoint).Define(
            stage, f'/robot/{child}_joint')
        joint.CreateBody0Rel().SetTargets(['/robot/shank'])
        joint.CreateBody1Rel().SetTargets([f'/robot/{child}'])
        z = {'foot': -.3, 'loose': -.6, 'hinged': -.9}[child]
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, z))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    stage.Save()
    before = path.read_text()
    stage, moved = physx_stage(path)
    assert list(moved) == ['foot'] and abs(moved['foot']['angle'] - .0012) < 1e-6
    cache = UsdGeom.XformCache()
    foot = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/robot/foot'))
    assert Gf.IsClose(foot, Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, -.3)), 1e-6)
    # Gaps too large to be rounding, and any joint angle, stay as authored; the file is unchanged.
    for name in ('loose', 'hinged'):
        assert not Gf.IsClose(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(f'/robot/{name}')),
                              Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, {'loose': -.6, 'hinged': -.9}[name])), 1e-6)
    assert path.read_text() == before


def drive(path, stiffness):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(path))
    api = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath('/robot/hinge'), 'angular')
    api.CreateTypeAttr('force')
    api.CreateStiffnessAttr(stiffness)
    api.CreateDampingAttr(0.1)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath('/robot'))
    stage.SetDefaultPrim(stage.GetPrimAtPath('/robot'))
    stage.Save()
    return path


def test_mjlab_usd_robot_uses_cached_conversion_and_usd_drives(tmp_path, monkeypatch):
    pytest.importorskip('mjlab')
    import math
    from mjlab.scene import Scene, SceneCfg
    import shared.asset_formats as formats
    from shared.backends.mjlab.assets import usd_robot
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    source = drive(write_usd(tmp_path / 'robot.usda'), 2.0)
    model = Scene(SceneCfg(num_envs=1, entities={'robot': usd_robot(source)}), device='cpu').compile()
    assert model.nu == 1 and model.actuator_gainprm[0, 0] == pytest.approx(2.0 * 180 / math.pi)
    cached = list((tmp_path / '.cache/asset_mjcf').iterdir())
    assert len(cached) == 1 and json.loads((cached[0] / 'ASSET.json').read_text())['alignment']['aligned']
    assert formats.mjcf_for(source) == cached[0] / 'mjcf/robot.xml'


def test_recorded_conversion_is_reused_until_source_changes(tmp_path, monkeypatch):
    import shared.asset_formats as formats
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    source = drive(write_usd(tmp_path / 'robot.usda'), 2.0)
    record = formats.convert(source, tmp_path, forms=('mjcf',))
    assert record['alignment']['aligned']
    assert formats.mjcf_for(source) == tmp_path / 'mjcf/robot.xml'
    drive(source, 3.0)
    with pytest.raises(ValueError, match='changed after'):
        formats.mjcf_for(source)


def test_mjlab_usd_object_takes_inertia_from_collision_geometry(tmp_path, monkeypatch):
    pytest.importorskip('mjlab')
    from pxr import Usd, UsdGeom, UsdPhysics
    from mjlab.scene import Scene, SceneCfg
    import shared.asset_formats as formats
    from shared.backends.mjlab.assets import usd_object
    monkeypatch.setattr(formats, 'ROOT', tmp_path)
    stage = Usd.Stage.CreateNew(str(tmp_path / 'cube.usda'))
    root = UsdGeom.Xform.Define(stage, '/cube')
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.2)
    cube = UsdGeom.Cube.Define(stage, '/cube/geometry')
    cube.CreateSizeAttr(0.06)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.Save()
    static = Scene(SceneCfg(num_envs=1, entities={'cube': usd_object(tmp_path / 'cube.usda')}), device='cpu').compile()
    free = Scene(SceneCfg(num_envs=1, entities={'cube': usd_object(tmp_path / 'cube.usda', mass=0.5)}),
                 device='cpu').compile()
    assert static.nq == 0 and static.body_inertia[-1] == pytest.approx([0.2 * 0.06 ** 2 / 6] * 3)
    assert free.nq == 7 and free.body_mass[-1] == pytest.approx(0.5)
    assert free.body_inertia[-1] == pytest.approx([0.5 * 0.06 ** 2 / 6] * 3)
    with pytest.raises(NotImplementedError, match='scale'):
        usd_object(tmp_path / 'cube.usda', scale=2.0)
