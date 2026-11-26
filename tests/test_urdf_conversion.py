"""The Kit-free URDF conversion keeps what a scene and the joint contract read.

`urdf-usd-converter` nests each link under its parent and welds the base to the world; these tests pin
the parts the pipeline depends on: body paths, joint order, drive handling, the free/fixed base and the
alignment against the MuJoCo-compiled MJCF of the same URDF.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip('pxr')
pytest.importorskip('urdf_usd_converter')

from pxr import Usd, UsdPhysics

from shared.asset_formats import compare, convert, convert_urdf, to_usd, usd_body_paths, usd_body_structure, \
    usd_joint_order
from shared.asset_import import DEFAULT_ACTUATORS, joint_stiffness, mimic_joints, undriven_joints

pytestmark = pytest.mark.group_import  # runs in the [import] dependency group

# Two links, a fixed tool frame (the converter keeps it as a plain Xform) and a mimic gripper finger.
URDF = '''<?xml version="1.0"?>
<robot name="tiny_arm">
  <link name="base">
    <inertial><origin xyz="0 0 0"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="arm">
    <inertial><origin xyz="0 0 0.1"/><mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
    <collision><origin xyz="0 0 0.1"/><geometry><box size="0.05 0.05 0.2"/></geometry></collision>
  </link>
  <link name="finger">
    <inertial><origin xyz="0 0 0.02"/><mass value="0.1"/>
      <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/></inertial>
    <collision><origin xyz="0 0 0.02"/><geometry><box size="0.02 0.02 0.04"/></geometry></collision>
  </link>
  <link name="tool"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><origin xyz="0 0 0.1"/><axis xyz="0 1 0"/>
    <limit lower="-1.5" upper="1.5" effort="10" velocity="2"/>
  </joint>
  <joint name="finger_joint" type="prismatic">
    <parent link="arm"/><child link="finger"/><origin xyz="0 0 0.2"/><axis xyz="1 0 0"/>
    <limit lower="0.0" upper="0.04" effort="5" velocity="0.5"/>
    <mimic joint="shoulder" multiplier="0.5" offset="0.01"/>
  </joint>
  <joint name="tool_joint" type="fixed">
    <parent link="finger"/><child link="tool"/><origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/>
  </joint>
</robot>
'''


def welded_to_world(usd):
    """Active fixed joints whose first body is the asset itself: the base is held to the world."""
    stage = Usd.Stage.Open(str(usd))
    world = str(stage.GetDefaultPrim().GetPath())
    return [str(prim.GetPath()) for prim in stage.Traverse()
            if prim.IsActive() and prim.IsA(UsdPhysics.FixedJoint)
            and [str(target) for target in UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()] == [world]]


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / 'tiny_arm.urdf'
    path.write_text(URDF)
    return path


def test_converted_urdf_keeps_the_body_and_joint_contract(urdf, tmp_path):
    usd, record = convert_urdf(urdf, tmp_path, 'tiny_arm', fix_base=False)
    assert usd == tmp_path / 'tiny_arm.usd' and usd.is_file()  # named after the asset, not the robot in the URDF
    assert record['tool'] == 'urdf-usd-converter' and record['fix_base'] is False
    assert json.loads((tmp_path / '.import.json').read_text()) == record

    assert usd_body_structure(usd) == 'articulation'
    assert usd_joint_order(usd) == ['shoulder', 'finger_joint']  # movable joints, fixed tool joint excluded
    # links keep the converter's own nesting, and a link with only a fixed joint stays addressable as a frame
    assert usd_body_paths(usd) == {'base': 'Geometry/base', 'arm': 'Geometry/base/arm',
                                   'finger': 'Geometry/base/arm/finger',
                                   'tool': 'Geometry/base/arm/finger/tool'}
    # no drive gains come from the converter: the import gives every joint the default PD instead
    assert not any(joint_stiffness(usd).values())
    assert sorted(undriven_joints(usd)) == ['finger_joint', 'shoulder']
    assert DEFAULT_ACTUATORS['.*']['stiffness'] > 0
    # couplings are read from the URDF, not from the USD
    assert mimic_joints(urdf) == {'finger_joint': {'leader': 'shoulder', 'ratio': 0.5, 'offset': 0.01}}


def test_free_base_deactivates_the_converter_weld(urdf, tmp_path):
    free, _ = to_usd(urdf, tmp_path / 'free', 'tiny_arm', fix_base=False)
    fixed, _ = to_usd(urdf, tmp_path / 'fixed', 'tiny_arm', fix_base=True)
    assert welded_to_world(free) == []  # the entity's own fix_root_link decides
    assert len(welded_to_world(fixed)) == 1
    for usd in (free, fixed):
        assert usd_joint_order(usd) == ['shoulder', 'finger_joint']
        assert usd_body_structure(usd) == 'articulation'


def test_converted_urdf_aligns_with_its_mjcf(urdf, tmp_path):
    asset = tmp_path / 'asset'
    record = convert(urdf, asset, name='tiny_arm')
    assert record['joint_order'] == ['shoulder', 'finger_joint']
    assert Path(record['formats']['usd']).suffix == '.usd'
    report = compare(asset / 'usd' / 'tiny_arm.usd', asset / 'mjcf' / 'tiny_arm.xml')
    assert report['aligned'], report['mismatches']
    assert not report['usd_only_bodies'] and not report['mjcf_only_bodies']


def test_link_kept_as_a_transform_is_a_robot_frame(urdf, tmp_path):
    """The converter leaves the massless tool link out of the articulation; ASSET.json records it as a frame
    on the body it is fixed to, so ``frame_pose('tool')`` and ``obs.<robot>.tool`` work as before."""
    from shared.assembly.layout import world_spec_from_config
    from shared.assembly.robot import robot_frames
    from shared.asset_formats import recorded_frames
    asset = tmp_path / 'asset'
    record = convert(urdf, asset, name='tiny_arm')
    turn = [pytest.approx(0.5 ** 0.5), 0.0, 0.0, pytest.approx(0.5 ** 0.5)]
    assert record['frames'] == {'tool': ['finger', [[0.0, 0.0, 0.1], turn]]}
    usd = asset / 'usd' / 'tiny_arm.usd'
    assert recorded_frames(usd)['tool'][0] == 'finger'
    settings = {'name': 'world', 'entities': [
        {'name': 'arm', 'kind': 'robot', 'factory': 'usd_object', 'parameters': {'usd_path': str(usd)},
         'frames': [{'name': 'palm', 'body': 'arm', 'pose': {'position': [0, 0, .1]}}]}]}
    frames = robot_frames(world_spec_from_config(settings, directory=tmp_path))['arm']
    assert frames['tool'] == ('finger', ((0.0, 0.0, 0.1), tuple(turn)))
    assert frames['palm'] == ('arm', ((0, 0, .1), (1., 0., 0., 0.)))  # the entity's own frames still add
