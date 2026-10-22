import json
from pathlib import Path

import pytest

pytest.importorskip('mujoco')
pytest.importorskip('pxr')

from shared.asset_formats import compare, urdf_to_mjcf, usd_frames_into_mjcf

INERTIAL = '<inertial><origin xyz="{com}"/><mass value="{mass}"/><inertia ixx="0.01" iyy="0.02" izz="0.03" ixy="0" ixz="0" iyz="0"/></inertial>'


def write_urdf(directory, com='0 0 0.1', mass=1.0, rpy='1.5707963 0 0'):
    (directory / 'part.obj').write_text(
        'mtllib part.mtl\nv 0 0 0\nv 0.1 0 0\nv 0 0.1 0\nv 0 0 0.1\n'
        'usemtl red\nf 1 2 3\nf 1 2 4\nf 1 3 4\nf 2 3 4\n')
    (directory / 'part.mtl').write_text('newmtl red\nKd 0.9 0.1 0.1\n')
    path = directory / 'robot.urdf'
    path.write_text(f'''<robot name="robot">
  <link name="base">{INERTIAL.format(com='0 0 0', mass=2.0)}
    <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
  <link name="arm">{INERTIAL.format(com=com, mass=mass)}
    <visual><geometry><mesh filename="part.obj"/></geometry></visual>
    <collision><geometry><mesh filename="part.obj"/></geometry></collision></link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 0.2" rpy="{rpy}"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
</robot>''')
    return path


def write_usd(path, com=(0.0, 0.0, 0.1), mass=1.0):
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
    joint = UsdPhysics.RevoluteJoint.Define(stage, '/robot/hinge')
    joint.CreateBody0Rel().SetTargets(['/robot/base'])
    joint.CreateBody1Rel().SetTargets(['/robot/arm'])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))
    joint.CreateLocalRot0Attr(Gf.Quatf(0.7071068, 0.7071068, 0, 0))
    joint.CreateAxisAttr('Z')
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
    assert len(geoms) == 1 and model.geom_rgba[geoms[0]] == pytest.approx([0.9, 0.1, 0.1, 1.0])
    assert sorted(path.name for path in (tmp_path / 'mjcf/assets').iterdir()) == ['part.stl']
    report = compare(write_usd(tmp_path / 'robot.usda'), mjcf)
    assert report['aligned'], report['mismatches']


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
