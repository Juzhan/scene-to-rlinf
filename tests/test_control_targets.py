"""Control targets are scene-owned: the property list and controller panel agree."""
from pathlib import Path

import pytest

from shared.assembly.control import toggle_target
from shared.assembly.editing import SceneDocument
from shared.assembly.properties import control_coverage, properties

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    return {'entities': [{'path': 'world/robot', 'native_key': 'robot',
                          'initial_display': {'joint_positions': {'hip': .1, 'knee': .2}}}],
            'catalog': {'observations': [{'id': 'robot.' + field, 'information': {'shape': [3]},
                'source_groups': {'default': {'sources': [{'path': 'world/robot'}]}}}
                for field in ('position', 'joint_position')]},
            'control': {'train': 'executor',
                        'components': {'planner': {'observations': ['robot.position']},
                                       'executor': {'observations': ['robot.joint_position']}},
                        'output': {}}}


def row(rows, reference):
    return next(item for item in rows if item.reference == reference)


def test_actuator_rows_publish_control_target_capability():
    rows = properties(snapshot(), entities={'world/robot': {}})
    joints = row(rows, 'robot.joint_position')
    assert joints.controllable == {'entity': 'robot', 'mode': 'joint_position', 'joint': None, 'joints': ['hip', 'knee']}
    assert joints.control == []
    assert joints.children[0].controllable['joint'] == 'hip'
    # Position rows are observations, not control targets.
    assert row(rows, 'robot.position').controllable is None


def test_toggle_covers_whole_collection_or_one_joint():
    # The whole collection is written out joint by joint from the measured names.
    assert toggle_target({}, 'robot', 'joint_position', None, True, ['hip', 'knee']) == {
        'robot': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['hip', 'knee']}}
    with pytest.raises(ValueError, match='inspect'):
        toggle_target({}, 'robot', 'joint_position', None, True)
    outputs = {'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['hip']}}
    assert toggle_target(outputs, 'robot', 'joint_position', 'hip', True) == {}
    assert toggle_target(outputs, 'robot', 'joint_position', 'knee', True) == {
        'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['hip', 'knee']}}
    # Removing the last explicit joint removes the binding itself.
    assert toggle_target(outputs, 'robot', 'joint_position', 'hip', False) == {'legs': None}
    # A wildcard target covers its joints as a whole and is not unbound per joint.
    wildcard = {'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['.*']}}
    with pytest.raises(ValueError, match='通配'):
        toggle_target(wildcard, 'robot', 'joint_position', 'hip', False)
    assert toggle_target(wildcard, 'robot', 'joint_position', None, False) == {'legs': None}
    # A different mode cannot also control a bound joint.
    with pytest.raises(ValueError, match='冲突'):
        toggle_target(outputs, 'robot', 'joint_effort', None, True, ['hip', 'knee'])
    # Selecting the whole collection extends an explicit target with the missing joints.
    assert toggle_target(outputs, 'robot', 'joint_position', None, True, ['hip', 'knee']) == {
        'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['hip', 'knee']}}
    assert toggle_target(outputs, 'robot', 'joint_effort', 'knee', True) == {
        'robot': {'type': 'joint_effort', 'asset_name': 'robot', 'joint_names': ['knee']}}


def test_property_rows_and_saved_scene_share_the_same_targets():
    settings = SceneDocument(ROOT / 'projects/assembly_go2_navigation/scene.yaml').snapshot()
    rows = properties(settings)
    assert {name for item in rows for name in item.control} == {'legs'}
    legs = row(rows, 'robot.joint_position')
    assert legs.control == ['legs']
    assert legs.controllable['mode'] == 'joint_position'
    assert all(item.control for item in legs.children)


def test_native_joint_patterns_cover_measured_joints():
    # Native control resolves joint_names as regular expressions, so ".*"
    # covers every measured joint of that asset and "panda_joint[1-7]" a range.
    wildcard = {'legs': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['.*']}}
    control = {'output': wildcard}
    assert control_coverage(control, 'robot', 'joint_position', 'FL_hip_joint') == ['legs']
    assert control_coverage(control, 'robot', 'joint_position') == ['legs']
    ranged = {'arm': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['panda_joint[1-7]']}}
    assert control_coverage({'output': ranged}, 'robot', 'joint_position', 'panda_joint3') == ['arm']
    assert control_coverage({'output': ranged}, 'robot', 'joint_position', 'panda_joint8') == []
    # Another asset or mode never borrows the binding.
    assert control_coverage(control, 'other', 'joint_position', 'FL_hip_joint') == []
    assert control_coverage(control, 'robot', 'joint_effort', 'FL_hip_joint') == []
    assert toggle_target(wildcard, 'robot', 'joint_position', 'FL_hip_joint', True) == {}


def test_abstract_entity_rows_are_observations_with_editable_size():
    data = SceneDocument(ROOT / 'projects/assembly_go2_navigation/scene.yaml').snapshot()
    rows = properties(data, entities={'world/arena': {'size': [4., 5.]}, 'world/destination': {}})
    size = row(rows, 'arena.size')
    assert (size.owner, size.value, size.edit) == ('world/arena', [4., 5.], ('world/arena', 'size'))
    goal = row(rows, 'destination.position')
    assert goal.observation == goal.task == 'destination.position'
    assert list(row(rows, 'destination.orientation').information['shape']) == [4]
    # A point has no size and no other entity's rows leak in.
    assert not any(item.reference == 'destination.size' for item in rows)
    assert not any(item.owner == 'world/robot' for item in rows)


def test_conflicts_use_measured_names_for_overlapping_regexes():
    from shared.assembly.control import validate_target_conflicts
    outputs = {'a': {'type': 'joint_position', 'asset_name': 'robot', 'joint_names': ['joint[1-2]']},
               'b': {'type': 'joint_effort', 'asset_name': 'robot', 'joint_names': ['joint[2-3]']}}
    with pytest.raises(ValueError, match='robot/joint2'):
        validate_target_conflicts(outputs, {'robot': ['joint1', 'joint2', 'joint3']})
    outputs['b']['asset_name'] = 'other_robot'
    validate_target_conflicts(outputs, {'robot': ['joint1', 'joint2', 'joint3']})


def test_conflicting_scene_edit_keeps_original_bindings():
    from copy import deepcopy
    doc = SceneDocument(ROOT / 'projects/assembly_go2_navigation/scene.yaml')
    before = deepcopy(doc.data['control']['output'])
    with pytest.raises(ValueError, match='冲突'):
        doc.set_action('torque', {'type': 'joint_effort', 'asset_name': 'robot', 'joint_names': ['FL_hip_joint']})
    assert doc.data['control']['output'] == before
    # Explicitly editing the existing target changes its mode without duplicating it.
    doc.set_action('legs', {**before['legs'], 'type': 'joint_velocity'})
    assert list(doc.data['control']['output']) == ['legs']


def test_new_target_defaults_to_free_joints_of_the_first_kind():
    from shared.assembly.control import default_target_joints
    arm = [f'joint{i}' for i in range(1, 7)]
    fingers = ['gripper_joint1', 'gripper_joint2']
    records = [{'name': name, 'type': 'revolute', 'limits': [-2.0, 2.0]} for name in arm]
    records += [{'name': name, 'type': 'prismatic', 'limits': [0.0, 0.05]} for name in fingers]
    assert default_target_joints(arm + fingers, set(), records) == arm
    assert default_target_joints(arm + fingers, set(arm), records) == fingers
    # Unlimited wheels are a kind of their own; without kinematic records every free joint is chosen.
    wheels = [{'name': 'wheel', 'type': 'revolute', 'limits': None}]
    assert default_target_joints(['hip', 'wheel'], set(), [{'name': 'hip', 'type': 'revolute', 'limits': [-1, 1]}, *wheels]) == ['hip']
    assert default_target_joints(arm + fingers, {'joint1'}) == arm[1:] + fingers


def test_retick_observation_returns_to_its_catalog_slot():
    from shared.assembly.control import observation_order
    catalog = ['a', 'b', 'c', 'd', 'e']
    # Saved order is kept as is (it is the model input order); unselected terms follow their catalog predecessor.
    assert observation_order(['d', 'a'], catalog) == ['d', 'e', 'a', 'b', 'c']
    assert observation_order(['c'], catalog) == ['a', 'b', 'c', 'd', 'e']
    assert observation_order(['a', 'c'], catalog) == ['a', 'b', 'c', 'd', 'e']
    selected = ['a', 'c']  # b was unticked and saved; ticking it again puts it between a and c.
    assert [key for key in observation_order(selected, catalog) if key in {*selected, 'b'}] == ['a', 'b', 'c']
    assert observation_order(['x', 'a'], catalog)[:2] == ['x', 'a']
