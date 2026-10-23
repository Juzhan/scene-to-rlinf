"""Browser editor: shared frame math, Blender-free previews and headless round trips."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from shared.assembly import frames
from shared.assembly.mesh_preview import TRIANGLE_BUDGET, simplify_manifest

ROOT = Path(__file__).resolve().parents[1]
EDITOR_PYTHON = ROOT / '.venv-editor/bin/python'


def test_pose_and_euler_round_trip():
    pose = {'position': [0.3, -0.2, 0.5], 'orientation': list(frames.matrix_quat(frames.euler_matrix((10, -20, 30))))}
    assert frames.same_pose(frames.matrix_pose(frames.pose_matrix(pose)), pose)
    assert np.allclose(frames.matrix_euler(frames.euler_matrix((10, -20, 30))), (10, -20, 30))


def test_fixed_connection_offsets_are_inverse():
    parent_pose = {'position': [0.1, 0.0, 0.06], 'orientation': [1.0, 0.0, 0.0, 0.0]}
    child_pose = {'position': [0.0, 0.0, 0.02], 'orientation': list(frames.matrix_quat(frames.euler_matrix((0, 0, 90))))}
    body_local = frames.transform((0.0, 0.0, 0.1))
    local = frames.fixed_child_local(parent_pose, child_pose, body_local)
    assert frames.same_pose(frames.fixed_parent_pose(local, body_local, child_pose), parent_pose)


def test_forward_kinematics_reproduces_exported_frames():
    exported = {'base': frames.IDENTITY_POSE,
                'link': {'position': [0.0, 0.0, 0.2], 'orientation': [1.0, 0.0, 0.0, 0.0]}}
    joint = {'name': 'j', 'parent': 'base', 'child': 'link', 'type': 'revolute', 'axis': 'Z', 'position': 0.0,
             'origin': {'position': [0.0, 0.0, 0.2], 'orientation': [1.0, 0.0, 0.0, 0.0]}}
    unchanged = frames.forward_kinematics(exported, [joint])
    assert np.allclose(unchanged['link'], frames.pose_matrix(exported['link']))
    turned = frames.forward_kinematics(exported, [joint], {'j': np.pi / 2})
    assert np.allclose(turned['link'][:3, 3], (0.0, 0.0, 0.2))
    assert np.allclose(turned['link'][:3, 0], (0.0, 1.0, 0.0))


def test_numpy_preview_keeps_source_parts_and_frames(tmp_path):
    grid = np.stack(np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200)), -1).reshape(-1, 2)
    vertices = np.c_[grid, np.zeros(len(grid))].tolist()
    index = np.arange(200 * 200).reshape(200, 200)
    faces = np.c_[index[:-1, :-1].ravel(), index[1:, :-1].ravel(), index[1:, 1:].ravel(), index[:-1, 1:].ravel()].tolist()
    exported = {'base': {'position': [0.1, 0.2, 0.3], 'orientation': [1.0, 0.0, 0.0, 0.0]}}
    source = tmp_path / 'asset.mesh.json.gz'
    with gzip.open(source, 'wt') as stream:
        json.dump({'parts': [{'name': 'big', 'vertices': vertices, 'faces': faces},
                             {'name': 'tiny', 'vertices': [[0, 0, 0], [1, 0, 0], [0, 1, 0]], 'faces': [[0, 1, 2]]}],
                   'frames': exported}, stream)
    original = source.read_bytes()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'entities': {'world/thing': {'key': 'k', 'path': str(source)}}}))
    simplify_manifest(manifest)
    record = json.loads(manifest.read_text())['entities']['world/thing']
    assert source.read_bytes() == original and record['key'] == 'k' and not record['exact']
    assert record['source_path'] == str(source) and record['triangle_count'] <= TRIANGLE_BUDGET
    with gzip.open(record['path'], 'rt') as stream:
        light = json.load(stream)
    assert [part['name'] for part in light['parts']] == ['big', 'tiny'] and light['frames'] == exported


@pytest.mark.skipif(not EDITOR_PYTHON.exists(), reason='.venv-editor (viser) unavailable')
def test_headless_editor_round_trip(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    scene = project / 'scene.yaml'
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
from tools.web_editor import Editor
from shared.assembly.editing import SceneDocument
scene = {str(scene)!r}
before = open(scene).read()
editor = Editor(scene, port=0, native_preview=False)
editor.save()
assert open(scene).read() == before, 'no-op save rewrote scene.yaml'
editor.select(('entity', 'world/table'))
editor.gizmo.position = (0.6, 0.1, 0.0)
editor._gizmo_moved(None)
editor.save()
pending = {{k: v for k, v in editor.collect().items() if k != 'revision' and v}}
assert not pending, pending
pose = SceneDocument(scene).entities['world/table']['pose']
assert [round(float(v), 6) for v in pose['position']] == [0.6, 0.1, 0.0], pose
editor.undo_edit()
pose = SceneDocument(scene).entities['world/table']['pose']
assert [round(float(v), 6) for v in pose['position']] != [0.6, 0.1, 0.0], 'undo kept the saved move'
assert open(scene).read() == before, 'undo did not restore scene.yaml'
editor.redo_edit()
assert [round(float(v), 6) for v in SceneDocument(scene).entities['world/table']['pose']['position']] == [0.6, 0.1, 0.0]
import tools.web_editor as W
assert W.observation_group('robot.target.linear_velocity', {{'group_label': 'robot'}}) == ('robot', 'target.linear_velocity')
assert W.observation_group('planner.output', {{}}, {{'planner': {{}}}}) == ('planner · 组件接口', 'output')
W.LIBRARY, W.THUMBNAILS = W.Path({str(tmp_path)!r}) / 'library.json', W.Path({str(tmp_path)!r}) / 'thumbs'
editor.start_thumbnails = lambda keys=None: None
assert editor.gallery_cards and editor.asset_choice in editor.gallery_cards
editor.import_path.value = 'projects/piper_phone_sim/assets/piper_x.urdf'
editor._import_asset()
library = W.load_library()
assert library[0]['factory'] == 'urdf_robot' and library[0]['parameters']['urdf_path'].endswith('piper_x.urdf'), library
assert editor.asset_choice == 'library:piper_x', editor.asset_choice
editor._add_asset()
assert any(item.get('factory') == 'urdf_robot' for item in SceneDocument(scene).snapshot()['entities'])
assert {{'abstract:point', 'abstract:box'}} <= set(editor.gallery_cards)
editor._choose_asset('abstract:box')
assert editor.asset_size.visible and not editor.asset_parameters.visible
editor.asset_position.value, editor.asset_size.value = (0.5, 0.0, 0.1), (0.2, 0.3, 0.4)
name = editor.asset_name.value
editor._add_asset()
added = SceneDocument(scene).entities['world/' + name]
assert added['kind'] == 'abstract' and added['shape'] == 'box' and added['size'] == [0.2, 0.3, 0.4], added
editor._choose_asset('abstract:point')
assert not editor.asset_size.visible
import json
W.TERRAINS = W.Path({str(tmp_path)!r}) / 'terrains'
(W.TERRAINS / 'stairs').mkdir(parents=True)
(W.TERRAINS / 'stairs' / 'ASSET.json').write_text(json.dumps({{
    'formats': {{'usd': 'assets/terrains/stairs/usd/stairs.usda'}},
    'terrain': {{'label': '台阶', 'size': 8.0, 'spawn_origin': [0.0, 0.0, 1.5]}}}}))
editor._refresh_assets()
tile = editor.gallery_entries['library:terrain_stairs']
assert tile['category'] == 'terrain' and tile['factory'] == 'usd_object' and tile['label'] == '台阶', tile
assert tile['parameters'] == {{'usd_path': 'assets/terrains/stairs/usd/stairs.usda'}}, tile
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_thumbnail_sketch_of_exported_mesh(tmp_path):
    mesh = tmp_path / 'part.mesh.json.gz'
    with gzip.open(mesh, 'wt') as stream:
        json.dump({'parts': [{'name': '/World/box/base/visual', 'vertices': [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                              'faces': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]}],
                   'frames': {'base': frames.IDENTITY_POSE}, 'joints': []}, stream)
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
from tools.web_editor import render_thumbnail, thumbnail_key
data = render_thumbnail({{'path': {str(mesh)!r}}})
assert data[:8] == b'\\x89PNG\\r\\n\\x1a\\n', data[:8]
assert thumbnail_key('box', {{}}) == 'box' != thumbnail_key('box', {{'size': [1, 1, 1]}})
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(not EDITOR_PYTHON.exists(), reason='.venv-editor (viser) unavailable')
def test_headless_picking_parent_and_axes(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import numpy as np
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
gui, handles = editor.server.gui, {{}}
for method in ('add_dropdown', 'add_button'):
    def wrap(label, *args, _original=getattr(gui, method), **kwargs):
        handles[label] = _original(label, *args, **kwargs)
        return handles[label]
    setattr(gui, method, wrap)
cube, table = ('entity', 'world/object'), ('entity', 'world/table')
# Overlapping bodies: clicking the selection again cycles to what lies behind it.
top = editor.info['world/object']['node'].world()[:3, 3] + (0, 0, 2)
first = editor.pick(top, (0, 0, -1))
assert first == cube, first
assert editor.pick(top, (0, 0, -1), cube) == table
assert editor.pick(top, (0, 0, -1), table) == cube
# Only the drawn gizmo handles block selection, not a sphere around the selection.
editor.select(table)
centre = np.asarray(editor.gizmo.position)
x_axis = W.F.quat_matrix(editor.gizmo.wxyz)[:3, 0]
assert editor._near_gizmo(centre + x_axis * W.GIZMO_SCALE * .8 + (0, 0, 2), (0, 0, -1))
assert not editor._near_gizmo(centre + (-.3, -.3, 2), (0, 0, -1))
# Choosing a parent only previews; 应用 writes it, keeps the world position, the gizmo and the open section.
editor.open_section = '父物体'
editor.select(cube)
world = editor.info['world/object']['node'].world()
assert not handles['应用'].visible and not handles['解除父物体（保留当前位置）'].visible
handles['父物体'].value = next(label for label in handles['父物体'].options if label.startswith('table'))
assert 'world/object' not in SceneDocument(editor.scene).snapshot().get('layout_parents', {{}}), 'applied before 应用'
assert handles['应用'].visible and editor.gizmo.visible
handles['应用'].value = True
assert SceneDocument(editor.scene).snapshot()['layout_parents']['world/object']['entity'] == 'world/table'
assert editor.selected == cube and editor.gizmo.visible and editor.open_section == '父物体', (editor.selected, editor.gizmo.visible, editor.open_section)
assert np.allclose(editor.info['world/object']['node'].world(), world, atol=1e-6)
assert not handles['应用'].visible and handles['解除父物体（保留当前位置）'].visible
handles['父物体'].value = W.NONE
assert SceneDocument(editor.scene).snapshot()['layout_parents']['world/object']['entity'] == 'world/table'
handles['应用'].value = True
assert 'world/object' not in SceneDocument(editor.scene).snapshot().get('layout_parents', {{}})
assert editor.open_section == '父物体'
# Deleting a parent asks first and keeps its children where they are.
assert editor.deletion_plan('world/robot')['dropped'] == ['world/ee_frame']  # A frame sensor needs its body.
editor.open_section = '父物体'
editor.select(cube)
handles['父物体'].value = next(label for label in handles['父物体'].options if label.startswith('table'))
handles['应用'].value = True
editor._delete('world/table')
assert 'world/table' in editor.items, 'deleted before confirmation'
handles['删除'].value = True
assert 'world/table' not in editor.items and 'world/object' in editor.items
assert not SceneDocument(editor.scene).snapshot().get('layout_parents')
assert np.allclose(editor.info['world/object']['node'].world(), world, atol=1e-6)
# Camera looking along world +Z with +X to the right: the X axis points right on screen.
svg = W.axes_svg((1.0, 0.0, 0.0, 0.0))
assert 'x2="26.0" y2="0.0" stroke="#e03131"' in svg, svg
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(not EDITOR_PYTHON.exists(), reason='.venv-editor (viser) unavailable')
def test_deleting_a_placement_region_keeps_dependants_static(tmp_path):
    project = tmp_path / 'navigation'
    shutil.copytree(ROOT / 'projects/planner_executor', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
gui, handles = editor.server.gui, {{}}
def wrap(label, *args, _original=gui.add_button, **kwargs):
    handles[label] = _original(label, *args, **kwargs)
    return handles[label]
gui.add_button = wrap
assert editor.items['world/destination']['reset']['region'] == 'world/arena'
assert editor.deletion_plan('world/arena')['unplaced'] == ['world/destination']
editor._delete('world/arena')
handles['删除'].value = True
entities = {{item['path']: item for item in SceneDocument(editor.scene).snapshot()['entities']}}
assert 'world/arena' not in entities and not entities['world/destination'].get('reset'), entities['world/destination']
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_control_tab_edits_targets_joint_by_joint(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
arm = [f'panda_joint{{i}}' for i in range(1, 8)]
assert editor.measured_joints()['robot'][:7] == arm, editor.measured_joints()
editor._toggle_joint('arm', 'panda_joint3', False)
assert editor.bindings()['arm']['joint_names'] == [j for j in arm if j != 'panda_joint3']
editor._toggle_joint('arm', 'panda_joint3', True)  # Back in articulation order.
assert editor.bindings()['arm']['joint_names'] == arm
editor._toggle_joint('arm', 'panda_finger_joint1', True)  # Owned by gripper: rejected, nothing staged.
assert editor.bindings()['arm']['joint_names'] == arm and 'error' in editor.status.content
editor._remove_target('gripper')
assert 'gripper' not in editor.bindings()
editor.target_asset.value, editor.target_name.value = 'robot', 'fingers'
editor._add_target()  # Only the joints no other target controls, each listed by name.
assert editor.bindings()['fingers'] == {{'type': 'joint_position', 'asset_name': 'robot',
    'joint_names': ['panda_finger_joint1', 'panda_finger_joint2']}}, editor.bindings()
editor.save()
output = SceneDocument(editor.scene).snapshot()['control']['output']
assert set(output) == {{'arm', 'fingers'}} and output['arm']['joint_names'] == arm, output
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_training_switch_and_module_pages(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
tabs = editor.observation_tabs
assert list(tabs) == ['planner', 'executor'] and editor.train_role == 'executor'
assert tabs['executor'].icon == 'flame' and tabs['planner'].icon == 'lock'
assert '"输出控制动作 · 训练中"' in editor.train_style.content and '"生成目标 · 固定"' in editor.train_style.content
assert 'button:nth-child(2){{--button-bg:var(--mantine-color-orange' in editor.train_style.content
# Planner sees no module interfaces, executor only the planner's; both pages list rows alike.
shown = {{role: [key for profile, key in editor.observation_boxes if profile == role] for role in tabs}}
assert not any(key.startswith(('planner.', 'executor.')) for key in shown['planner']), shown['planner']
assert 'planner.output' in shown['executor'] and not any(key.startswith('executor.') for key in shown['executor'])
common = [key for key in shown['executor'] if key in shown['planner']]
assert common == [key for key in shown['planner'] if key in shown['executor']]
editor._set_train('planner')
editor._set_train('planner')  # Already trained: no second undo step.
assert tabs['planner'].icon == 'flame' and '"生成目标 · 训练中"' in editor.train_style.content
assert editor.collect()['control_train'] == 'planner'
editor.undo_edit()
assert editor.train_role == 'executor' and 'control_train' not in editor.collect()
assert editor.observation_tabs['executor'].icon == 'flame'
editor.redo_edit()
# Each module page edits only that module's inputs.
saved = SceneDocument(editor.scene).snapshot()['control']['components']
key = next(key for key in editor.observation_order['planner'] if key not in saved['planner']['observations']
           and not editor.observation_boxes[('planner', key)].disabled)
editor.observation_boxes[('planner', key)].value = True
# Group folders holding a ticked box are highlighted client-side via :checked.
uuid = editor.observation_boxes[('planner', key)]._impl.uuid
assert any(f'input[id="{{uuid}}"]:checked' in getattr(item, 'content', '') for item in editor.observation_items)
assert editor.collect()['observations'] == {{'planner': saved['planner']['observations'] + [key]}}
editor.period_fields['planner'].value = '0.015'  # Not a multiple of dt 0.01: rejected on save.
editor.save()
assert 'error' in editor.status.content
editor.period_fields['planner'].value = '0.1'
assert editor.collect()['components'] == {{'planner': {{'period': 0.1}}}}
editor.dimension_fields['planner'].value = 'auto'  # Only executor may resolve its dimension.
editor.save()
assert 'error' in editor.status.content
editor.dimension_fields['planner'].value = '6'
assert editor.collect()['components'] == {{'planner': {{'period': 0.1, 'dimension': 6}}}}
editor.save()
saved = SceneDocument(editor.scene).snapshot()['control']
assert saved['train'] == 'planner' and key in saved['components']['planner']['observations']
assert saved['components']['planner']['period'] == 0.1 and saved['components']['planner']['dimension'] == 6
assert key not in saved['components']['executor']['observations']
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(not EDITOR_PYTHON.exists(), reason='.venv-editor (viser) unavailable')
def test_typed_camera_parameters(tmp_path):
    project = tmp_path / 'piper_pick'
    shutil.copytree(ROOT / 'projects/piper_pick', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
from shared.assembly.catalog import describe_scene
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
editor.select(('entity', 'world/wrist_camera'))
fields = editor.parameter_fields
assert set(fields) == {{'width', 'height', 'data_types', 'focal_length', 'clipping_range', 'depth_clipping'}}, fields
fields['width'].value = 64
fields['data_types']['distance_to_image_plane'].value = True
assert editor.camera_optics(editor.items['world/wrist_camera'])[:2] == (64.0, 120.0)
editor.undo_edit()  # One step per field: only the depth output reverts.
assert editor.collect()['entities'][0]['changes']['parameters'] == {{'width': 64}}, editor.collect()
editor.select(('entity', 'world/wrist_camera'))
editor.parameter_fields['data_types']['distance_to_image_plane'].value = True
editor.save()
parameters = SceneDocument(editor.scene).entities['world/wrist_camera']['parameters']
assert parameters == {{'width': 64, 'data_types': ['rgb', 'distance_to_image_plane']}}, parameters
depth = next(e for e in describe_scene(editor.scene)['observations'] if e['id'] == 'wrist_camera.distance_to_image_plane')
assert list(depth['information']['shape']) == [120, 64, 1]
editor.select(('entity', 'world/wrist_camera'))
assert editor.parameter_fields['data_types']['distance_to_image_plane'].value
editor.parameter_fields['data_types']['rgb'].value = False
editor.parameter_fields['data_types']['distance_to_image_plane'].value = False
editor.save()  # No output left: rejected with an explicit message, file unchanged.
assert 'error' in editor.status.content and 'data_types' in editor.status.content, editor.status.content
assert SceneDocument(editor.scene).entities['world/wrist_camera']['parameters'] == parameters
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
