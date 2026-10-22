from shared.assembly.properties import properties


def snapshot():
    return {'entities': [{'path': 'world/robot', 'native_key': 'robot', 'initial_display': {'joint_positions': {'hip': .1, 'knee': .2}}}],
            'catalog': {'observations': [{'id': 'robot.' + field, 'information': {'shape': [3]},
                'source_groups': {'default': {'sources': [{'path': 'world/robot'}]}}} for field in ('position', 'joint_position')]}}


def test_shared_references_selection_and_real_children():
    rows = properties(snapshot(), entities={'world/robot': {'position': [1, 2, 3]}})
    position, joints = rows[:2]
    assert position.name == 'position'
    assert position.reference == position.observation == position.task == 'robot.position'
    assert position.value == [1, 2, 3] and position.edit == ('world/robot', 'position')
    assert [p.name for p in joints.children] == ['hip', 'knee']
    assert joints.children[1].task == 'robot.joint_position[..., 1]'
    assert joints.children[1].value == .2
    assert joints.children[1].observation is None  # No invented component reader.


def test_pending_variables_share_directory_without_becoming_observations():
    row = properties(snapshot(), variables={'distance': {'expr': '0.0'}})[-1]
    assert row.task == 'distance' and row.owner == '共享变量'
    assert row.observation is None and row.value_source == 'computed'


def test_current_project_properties_are_task_references():
    from pathlib import Path
    from shared.assembly.editing import SceneDocument
    snapshot = SceneDocument(Path(__file__).resolve().parents[1] / 'projects/assembly_go2/scene.yaml').snapshot()
    row = next(p for p in properties(snapshot) if p.reference == 'robot.position')
    assert row.task == row.observation == 'robot.position'
    assert row.owner == 'world/robot' and row.name == 'position'


def test_removed_live_entity_is_not_offered_by_other_views():
    rows = properties(snapshot(), entities={})
    assert rows == []


def test_saved_variables_have_one_shared_observation_and_task_entry():
    data = snapshot()
    data['variables'] = {'distance': {'expr': '0.0'}}
    data['catalog']['observations'].append({'id': 'distance', 'available': True})
    rows = [r for r in properties(data) if r.reference == 'distance']
    assert len(rows) == 1
    assert rows[0].task == 'distance'
    assert rows[0].observation is None
    assert rows[0].owner == '共享变量'
    assert not any(r.reference == 'distance' for r in properties(data, variables={}))
