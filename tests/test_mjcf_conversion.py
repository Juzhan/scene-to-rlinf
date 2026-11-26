"""The Kit-free MJCF conversion keeps what a scene and the joint contract read.

`mujoco-usd-converter` nests each body under its parent, welds the root body to the world, and writes a fixed
joint for a body that MuJoCo attaches without one; these tests pin the parts the pipeline depends on.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip('pxr')
pytest.importorskip('mujoco_usd_converter')

from pxr import Usd, UsdPhysics

from shared.asset_formats import compare, convert, convert_mjcf, to_usd, usd_body_paths, usd_body_structure, \
    usd_joint_order
from shared.asset_import import DEFAULT_ACTUATORS, joint_stiffness, undriven_joints

pytestmark = pytest.mark.group_usd  # runs in the [usd] dependency group (its own mujoco-core)

# A hinge, a slide, and a tool body that MuJoCo attaches without a joint (the converter welds it).
MJCF = '''<mujoco model="tiny_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.2">
      <inertial pos="0 0 0" mass="1.0" diaginertia="0.01 0.01 0.01"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="arm" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <inertial pos="0 0 0.1" mass="0.5" diaginertia="0.01 0.01 0.01"/>
        <geom type="box" pos="0 0 0.1" size="0.025 0.025 0.1"/>
        <body name="finger" pos="0 0 0.2">
          <joint name="finger_joint" type="slide" axis="1 0 0" range="0 0.04"/>
          <inertial pos="0 0 0.02" mass="0.1" diaginertia="0.001 0.001 0.001"/>
          <geom type="box" pos="0 0 0.02" size="0.01 0.01 0.02"/>
          <body name="tool" pos="0 0 0.1">
            <inertial pos="0 0 0" mass="0.02" diaginertia="0.0001 0.0001 0.0001"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
'''


def welded_to_world(usd):
    """Active fixed joints whose first body is the asset itself: the base is held to the world."""
    stage = Usd.Stage.Open(str(usd))
    world = str(stage.GetDefaultPrim().GetPath())
    return [str(prim.GetPath()) for prim in stage.Traverse()
            if prim.IsActive() and prim.IsA(UsdPhysics.FixedJoint)
            and [str(target) for target in UsdPhysics.Joint(prim).GetBody0Rel().GetTargets()] == [world]]


@pytest.fixture
def mjcf(tmp_path):
    path = tmp_path / 'tiny_arm.xml'
    path.write_text(MJCF)
    return path


def test_converted_mjcf_keeps_the_body_and_joint_contract(mjcf, tmp_path):
    usd, record = convert_mjcf(mjcf, tmp_path, 'tiny_arm', fix_base=False)
    assert usd == tmp_path / 'tiny_arm.usd' and usd.is_file()  # named after the asset, not the model in the MJCF
    assert record['tool'] == 'mujoco-usd-converter' and record['fix_base'] is False
    assert json.loads((tmp_path / '.import.json').read_text()) == record

    assert usd_body_structure(usd) == 'articulation'
    assert usd_joint_order(usd) == ['shoulder', 'finger_joint']  # movable joints; the welded tool body has none
    # bodies keep the converter's own nesting, and the welded tool body is still addressable
    assert usd_body_paths(usd) == {'base': 'Geometry/base', 'arm': 'Geometry/base/arm',
                                   'finger': 'Geometry/base/arm/finger',
                                   'tool': 'Geometry/base/arm/finger/tool'}
    # no drive gains come from the converter: the import gives every joint the default PD instead
    assert not any(joint_stiffness(usd).values())
    assert sorted(undriven_joints(usd)) == ['finger_joint', 'shoulder']
    assert DEFAULT_ACTUATORS['.*']['stiffness'] > 0


def test_free_mjcf_base_deactivates_the_converter_weld(mjcf, tmp_path):
    free, _ = to_usd(mjcf, tmp_path / 'free', 'tiny_arm', fix_base=False)
    fixed, _ = to_usd(mjcf, tmp_path / 'fixed', 'tiny_arm', fix_base=True)
    assert welded_to_world(free) == []  # the entity's own fix_root_link decides
    assert len(welded_to_world(fixed)) == 1
    for usd in (free, fixed):
        assert usd_joint_order(usd) == ['shoulder', 'finger_joint']
        assert usd_body_structure(usd) == 'articulation'


def test_converted_mjcf_aligns_with_its_source(mjcf, tmp_path):
    asset = tmp_path / 'asset'
    record = convert(mjcf, asset, name='tiny_arm')
    assert record['joint_order'] == ['shoulder', 'finger_joint']
    report = compare(asset / 'usd' / 'tiny_arm.usd', mjcf)
    assert report['aligned'], report['mismatches']
    assert not report['usd_only_bodies'] and not report['mjcf_only_bodies']
