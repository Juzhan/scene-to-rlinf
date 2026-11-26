"""Browser editor: shared frame math, Blender-free previews and headless round trips."""
import gzip
from importlib.util import find_spec
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from shared.assembly import frames
from shared.assembly.mesh_preview import TRIANGLE_BUDGET, simplify_manifest

ROOT = Path(__file__).resolve().parents[1]
# The editor shares the simulator environment; headless round trips run under this interpreter.
EDITOR_PYTHON = Path(sys.executable)


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


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_newton_style_floor_tiles_the_square_and_sky_fades_down():
    from tools.web_editor import checker_floor, sky_gradient
    (light, light_faces), (dark, dark_faces) = checker_floor(size=4.0, cell=1.0)
    assert len(light_faces) == len(dark_faces) == 16  # 8 cells of each colour, two triangles each
    both = np.vstack([light, dark])
    assert both[:, :2].min() == -2.0 and both[:, :2].max() == 2.0 and not both[:, 2].any()
    sky = sky_gradient((68, 161, 255), (40, 44, 55), height=8)
    assert sky.shape == (8, 4, 3) and tuple(sky[0, 0]) == (68, 161, 255) and tuple(sky[-1, 0]) == (40, 44, 55)


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_headless_editor_round_trip(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    # An older project written for Isaac Lab 2 (urdf_robot, native presets): it must run on that backend.
    recipe = project / 'train.yaml'
    source = recipe.read_text()
    if 'backend:' not in source:
        recipe.write_text(source.replace('    init_params:\n', '    init_params:\n      backend: isaac\n', 1))
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
saved = open(scene).read()
editor.undo_edit()  # Undo edits the working copy; scene.yaml keeps what was saved.
pose = SceneDocument(editor.scene).entities['world/table']['pose']
assert [round(float(v), 6) for v in pose['position']] != [0.6, 0.1, 0.0], 'undo kept the saved move'
assert editor.scene.read_text() == before, 'undo did not restore the working copy'
assert open(scene).read() == saved and editor.unsaved()
editor.redo_edit()
assert [round(float(v), 6) for v in SceneDocument(editor.scene).entities['world/table']['pose']['position']] == [0.6, 0.1, 0.0]
assert not editor.unsaved()
import tools.web_editor as W
assert W.observation_group('robot.target.linear_velocity', {{'group_label': 'robot'}}) == ('robot', 'target.linear_velocity')
assert W.observation_group('planner.output', {{}}, {{'planner': {{}}}}) == ('planner · 组件接口', 'output')
W.LIBRARY, W.THUMBNAILS = W.Path({str(tmp_path)!r}) / 'library.json', W.Path({str(tmp_path)!r}) / 'thumbs'
editor.start_thumbnails = lambda keys=None: None
assert editor.shelf_items and editor.asset_choice is None  # nothing chosen until a card is picked
editor.import_path.value = 'projects/piper_phone_sim/assets/piper_x.urdf'
editor._import_asset()
library = W.load_library()
assert library[0]['factory'] == 'urdf_robot' and library[0]['parameters']['urdf_path'].endswith('piper_x.urdf'), library
assert editor.asset_choice == 'library:piper_x', editor.asset_choice
editor._add_asset()
assert any(item.get('factory') == 'urdf_robot' for item in SceneDocument(editor.scene).snapshot()['entities'])
assert open(scene).read() == saved and editor.unsaved(), 'adding an entity wrote scene.yaml before save'
assert {{'abstract:point', 'abstract:box'}} <= {{item['id'] for item in editor.shelf_items}}
editor._choose_asset('abstract:box')
assert editor.asset_size.visible and not editor.asset_parameters.visible
editor.asset_position.value, editor.asset_size.value = (0.5, 0.0, 0.1), (0.2, 0.3, 0.4)
name = editor.entity_name
editor._add_asset()
added = SceneDocument(editor.scene).entities['world/' + name]
assert added['kind'] == 'abstract' and added['shape'] == 'box' and added['size'] == [0.2, 0.3, 0.4], added
editor._remove('world/' + name, editor.deletion_plan('world/' + name))
editor._refresh_native()
assert open(scene).read() == saved, 'delete or mesh refresh wrote scene.yaml before save'
editor.save()
assert open(scene).read() == editor.scene.read_text() != saved and not editor.unsaved()
assert any(item.get('factory') == 'urdf_robot' for item in SceneDocument(scene).snapshot()['entities'])
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


def test_urdf_quick_import_names_the_conversion_on_backends_without_it(tmp_path):
    """Only the isaac backend builds a scene from a URDF; isaaclab3/mjlab need the converted USD."""
    from tools.web_editor import Editor

    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities: []\n')
    (tmp_path / 'train.yaml').write_text(
        'defaults: [state_ppo, _self_]\nenv:\n  train:\n    init_params:\n      backend: isaaclab3\n')
    editor = Editor(scene, port=0, native_preview=False)
    try:
        editor.import_path.value = 'projects/piper_phone_sim/assets/piper_x.urdf'
        editor._import_asset()  # guarded: the page shows the reason instead of raising
        assert editor.messages and editor.messages[-1]['error']
        assert '不能直接用 URDF 生成实体' in editor.messages[-1]['text']
        assert 'asset import' in editor.messages[-1]['text']
    finally:
        editor.server.stop()


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


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
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
unit = editor._gizmo_unit(centre + (0, 0, 2))  # the fixed-size gizmo's world length seen from 2 m above
assert editor._near_gizmo(centre + x_axis * unit * .8 + (0, 0, 2), (0, 0, -1))
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
# The floor is the ground plane entity's; the faint reference grid shows only while no ground is shown.
floor = ('entity', 'world/floor')
assert editor.info['world/floor']['node'].shape['type'] == 'checker' and not editor.floor_grid.visible
editor.toggle_hidden(floor)
assert editor.floor_grid.visible
editor.toggle_hidden(floor)
assert not editor.floor_grid.visible
# The app asks what a deletion changes, shows it in its own dialog and deletes only on confirmation.
editor.select(floor)
plan = editor.ui_get('delete', {{}})
assert plan['paths'] == ['world/floor'] and plan['lines'][-1] == '可用 Ctrl-Z 撤回。', plan
assert 'world/floor' in editor.items
editor.ui_post('delete', {{'paths': plan['paths']}})
assert 'world/floor' not in editor.items and editor.floor_grid.visible
# Camera looking along world +Z with +X to the right: the X axis points right on screen.
svg = W.axes_svg((1.0, 0.0, 0.0, 0.0))
assert 'x2="26.0" y2="0.0" stroke="#e03131"' in svg, svg
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_scene_clicks_with_ctrl_or_shift_extend_the_selection(tmp_path):
    """Ctrl/Cmd-click in the 3D view toggles and Shift-click adds, building the same selection the outliner
    rows do, so the gizmo, hiding and deleting act on all of them."""
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
from types import SimpleNamespace
sys.path.insert(0, {str(ROOT)!r})
import numpy as np
import tools.web_editor as W
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
robot, table = ('entity', 'world/robot'), ('entity', 'world/table')
click_cbs = {{entry.modifier: entry.callback for entry in editor.server.scene._scene_pointer_cb
             if entry.event_type == 'click'}}
assert set(click_cbs) == {{None, 'cmd/ctrl', 'shift'}}, sorted(click_cbs, key=str)


def click(modifier, origin, direction=(0, 0, -1)):
    """A click as viser delivers it: the modifier decides which registered callback fires."""
    event = SimpleNamespace(ray_origin=origin, ray_direction=direction, modifier=modifier, client=None, client_id=0,
                            screen_pos=((0.5, 0.5),))
    click_cbs[modifier](event)


above_robot = (0., 0., 2.)
above_table = np.asarray(editor.info['world/table']['node'].world())[:3, 3] + (0.3, 0, 2)
sky = (0., 0., 20.)
assert editor.pick(above_robot, (0, 0, -1)) == robot and editor.pick(above_table, (0, 0, -1)) == table
click(None, above_table)
assert editor.selection() == [table] and editor.selected == table
click('cmd/ctrl', above_robot)  # the robot joins; the table keeps its highlight
assert set(editor.selection()) == {{robot, table}} and editor.selected == robot
rows = {{row['id']: row for row in editor.outliner_get('state', {{}})['rows']}}
assert rows['entity:world/robot']['selected'] and rows['entity:world/table']['selected']
click('shift', above_table)  # already selected: Shift keeps it and only makes it active
assert set(editor.selection()) == {{robot, table}} and editor.selected == table
click('cmd/ctrl', above_robot)  # Ctrl-clicking a selected object drops it
assert editor.selection() == [table]
click('cmd/ctrl', sky, (0, 0, 1))  # a modifier click on empty space keeps the selection
assert editor.selection() == [table]
click('shift', above_robot)  # Shift-click adds
assert set(editor.selection()) == {{robot, table}} and editor.selected == robot
assert editor._near_gizmo(above_robot, (0, 0, -1))  # the gizmo sits here, so a click on it changes nothing
click('cmd/ctrl', above_robot)
assert set(editor.selection()) == {{robot, table}}
click(None, sky, (0, 0, 1))  # a plain click on empty space clears it
assert editor.selection() == [] and editor.selected is None
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_placed_entities_are_set_down_on_the_surface_under_them(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import numpy as np
import tools.web_editor as W
from tools.web_editor import Editor
editor = Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
entry = next(item for item in editor.snapshot['catalog']['assets'] if item['id'] == 'dex_cube')
def place(name, position):
    editor._add_entity({{'name': name, 'factory': 'dex_cube', 'kind': 'entity', 'parameters': {{}},
                        'pose': {{'position': position, 'orientation': [1, 0, 0, 0]}}}}, entry, rest=True)
    return editor._world_points('world/' + name)
# Sunk under the floor (here 1.05 m below the table top) or in the air, away from the table: its lowest point
# ends REST_CLEARANCE above the floor.
floor = editor.surface_point([-1.5, 1.5, 3.0], [0, 0, -1])[2]
assert floor < -1.0
for name, z in (('sunk', -1.5), ('floating', 1.0)):
    lowest = place(name, [-1.5, 1.5, z])[:, 2].min()
    assert abs(lowest - (floor + W.REST_CLEARANCE)) < 1e-6, (name, lowest, floor)
# Over the table, sunk into its top: set on it, as the view's own ray finds it.
top = editor.surface_point([0.5, 0.3, 3.0], [0, 0, -1])[2]  # clear of the scene's own cube at (0.5, 0)
assert top > floor + 0.5, (top, floor)  # the table top, about 1 m over the floor
lowest = place('on_table', [0.5, 0.3, -0.01])[:, 2].min()  # cutting into the table top: lifted onto it
assert abs(lowest - (top + W.REST_CLEARANCE)) < 1e-3, (lowest, top)
# Pressing on an object turns the view; only the gizmo moves it. Moved level with snapping, from the floor
# onto the table, it slides up onto the top; End sets a lifted one down.
from types import SimpleNamespace
cube = ('entity', 'world/sunk')
assert not editor._body_draggable('world/sunk')
editor.select(cube)
level = editor.gizmo.position[2]
for phase in ('update', 'update', 'end'):
    editor.gizmo.position = (0.5, 0.35, level)
    editor._gizmo_moved(SimpleNamespace(phase=phase))
assert abs(editor._world_points('world/sunk')[:, 2].min() - (top + W.REST_CLEARANCE)) < 1e-3
editor.ui_post('pose', {{'position': [0.5, 0.35, 0.8], 'rotation': [0, 0, 0]}})
editor.ui_post('drop', {{}})
assert abs(editor._world_points('world/sunk')[:, 2].min() - (top + W.REST_CLEARANCE)) < 1e-3
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
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


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_deleting_a_ray_target_leaves_the_scanner_to_be_retargeted(tmp_path):
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
texts = []
def note(text, *args, _original=gui.add_markdown, **kwargs):
    texts.append(text)
    return _original(text, *args, **kwargs)
gui.add_markdown = note
assert editor.backend_problem is None, editor.backend_problem
editor._delete('world/robot')  # The dialog names the targets and observations that go with it.
modal = next(text for text in texts if text.startswith('删除 **robot**'))
assert '不再受策略控制：`robot`' in modal and '`robot.joint_position`' in modal, modal
assert editor.deletion_plan('world/ground')['retargeted'] == ['world/height_scanner']
editor._delete('world/ground')
handles['删除'].value = True
entities = {{item['path']: item for item in SceneDocument(editor.scene).snapshot()['entities']}}
assert 'world/ground' not in entities and entities['world/height_scanner']['meshes'] == [], entities
# Saved, but the project's isaaclab3 backend has no global ground for the rays: the page says so.
assert 'scope: global' in editor.backend_problem and 'scope: global' in editor.status.content, editor.backend_problem
editor.select('world/height_scanner')
editor.pending['meshes']['world/height_scanner'] = ['world/robot']
editor.save()
entities = {{item['path']: item for item in SceneDocument(editor.scene).snapshot()['entities']}}
assert entities['world/height_scanner']['meshes'] == ['world/robot'], entities['world/height_scanner']
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_policy_lens_sets_robot_modes_and_lists_joints(tmp_path):
    """Robots are driven as a whole: the policy lens tints them, a robot's mode is saved as an exception in
    control.modes, and its movable joints list in the target order with fixed links folded away."""
    project = tmp_path / 'navigation'
    shutil.copytree(ROOT / 'projects/planner_executor', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
assert editor.robot_modes() == {{'robot': 'joint_position'}}
editor.outliner_post('lens', {{'lens': 'observe', 'role': 'executor'}})  # what the executor reads and drives
assert editor.tints == {{'world/robot': W.LENS_COLORS['executor']}}
state = editor.outliner_post('mode', {{'id': 'entity:world/robot', 'mode': 'joint_velocity'}})
assert [row['mode'] for row in state['rows'] if row['id'] == 'entity:world/robot'] == ['joint_velocity']
assert editor.collect()['control_modes'] == {{'robot': 'joint_velocity'}}
editor.save()
assert SceneDocument(editor.scene).snapshot()['control']['modes'] == {{'robot': 'joint_velocity'}}
editor.outliner_post('mode', {{'id': 'entity:world/robot', 'mode': 'none'}})
assert editor.tints.get('world/robot') == (W.LENS_COLORS['executor'] if editor.observation_uses().get('robot', {{}}).get('executor') else None)
editor.outliner_post('mode', {{'id': 'entity:world/robot', 'mode': 'joint_position'}})  # the default again
editor.save()
assert 'modes' not in SceneDocument(editor.scene).snapshot()['control']
# Joint rows: a fixed link folds into its parent, a serial chain stays at one depth, a branch indents.
joints = [dict(name='j1', parent='base', child='l1', type='revolute'), dict(name='mount', parent='l1', child='hand', type='fixed'),
          dict(name='f1', parent='hand', child='finger1', type='prismatic'), dict(name='f2', parent='hand', child='finger2', type='prismatic'),
          dict(name='j2', parent='l1', child='l2', type='revolute')]
editor.info['world/robot']['kinematics'] = {{'frames': {{}}, 'joints': joints}}
editor.initial_display = lambda path: {{'joint_positions': {{'j1': 0, 'j2': 0, 'f1': 0, 'f2': 0}}}}
assert [(row['name'], row['depth'], row['index']) for row in editor.joint_rows('world/robot')] == \
    [('j1', 0, 0), ('f1', 1, 2), ('f2', 1, 3), ('j2', 1, 1)]
editor.select_joint('world/robot', 'j2')
assert editor.panel_context()['joint']['name'] == 'j2' and editor.selected == ('entity', 'world/robot')
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_observe_lens_and_component_settings(tmp_path):
    project = tmp_path / 'lift'
    shutil.copytree(ROOT / 'projects/assembly_lift', project, ignore=shutil.ignore_patterns('__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
assert list(editor.observation_order) == ['planner', 'executor'] and editor.train_role == 'executor'
# control.output (the older form) lists its targets per asset; their modes are not switched per robot.
assert editor.robot_modes() == {{'robot': 'joint_position'}}
try:
    editor.set_robot_mode.__wrapped__(editor, 'robot', 'joint_velocity')
    raise AssertionError('a listed control.output must not take modes')
except ValueError:
    pass
# Planner sees no component interfaces, executor only the planner's.
groups = {{role: [g['key'] for g in editor.observation_groups(role)] for role in editor.observation_order}}
assert not {{'planner', 'executor'}} & set(groups['planner']) and 'executor' not in groups['executor'], groups
assert 'planner' in groups['executor']
state = editor.outliner_post('train', {{'role': 'planner'}})
assert state['train'] == 'planner' and editor.collect()['control_train'] == 'planner'
editor.undo_edit()
assert editor.train_role == 'executor' and 'control_train' not in editor.collect()
editor.redo_edit()
saved = SceneDocument(editor.scene).snapshot()['control']['components']
key = next(row['key'] for group in editor.observation_groups('planner') for row in group['rows']
           if not row['enabled'] and row['available'])
editor.outliner_post('lens', {{'lens': 'observe', 'role': 'planner'}})
editor.outliner_post('observe', {{'role': 'planner', 'key': key, 'on': True}})
native = key.partition('.')[0]
path = editor.native_path(native)
assert path is None or editor.tints.get(path) == W.LENS_COLORS['planner']
ticked = [item for item in editor.observation_order['planner'] if item in {{key, *saved['planner']['observations']}}]
assert editor.collect()['observations'] == {{'planner': ticked}}
editor.outliner_post('component', {{'role': 'planner', 'field': 'period', 'value': '0.015'}})  # not a multiple of dt
editor.save()
assert 'error' in editor.status.content
editor.outliner_post('component', {{'role': 'planner', 'field': 'period', 'value': '0.1'}})
editor.outliner_post('component', {{'role': 'planner', 'field': 'dimension', 'value': '6'}})
assert editor.collect()['components'] == {{'planner': {{'period': 0.1, 'dimension': 6}}}}
assert [c['period'] for c in editor.panel_context()['components'] if c['role'] == 'planner'] == ['0.1']
editor.save()
saved = SceneDocument(editor.scene).snapshot()['control']
assert saved['train'] == 'planner' and key in saved['components']['planner']['observations']
assert saved['components']['planner']['period'] == 0.1 and saved['components']['planner']['dimension'] == 6
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_shelf_sensors_arrive_mounted_and_ray_sensors_keep_the_ground(tmp_path):
    project = tmp_path / 'navigation'
    shutil.copytree(ROOT / 'projects/planner_executor', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
import shared.assembly.visuals as V
V.SHARED_CACHE = V.Path({str(tmp_path / 'cold_cache')!r})  # Ignore meshes cached on this machine.
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
texts = []
def note(text, *args, _original=editor.server.gui.add_markdown, **kwargs):
    texts.append(text)
    return _original(text, *args, **kwargs)
editor.server.gui.add_markdown = note
def entities():
    return {{item['path']: item for item in SceneDocument(editor.scene).snapshot()['entities']}}
editor.initial_display = lambda path: {{}}  # Link names come from the native export, not done yet.
editor._choose_asset('contact_sensor')
editor._add_asset()
assert '刷新原生网格' in editor.status.content and 'world/contact_sensor' not in entities(), editor.status.content
editor.initial_display = lambda path: {{'root_body': 'base', 'body_names': ['base', 'FL_foot', 'FR_foot']}} \
    if path == 'world/robot' else {{}}
for factory in ('contact_sensor', 'frame_transformer', 'height_scanner', 'camera'):
    editor._choose_asset(factory)
    assert editor.asset_mount.visible and editor.asset_mount.value.startswith('robot'), editor.asset_mount.options
    # isaaclab3: a camera may also stand fixed in the world, every other sensor follows a body
    assert (W.FREE in editor.asset_mount.options) == (factory == 'camera'), editor.asset_mount.options
    editor._add_asset()
    assert editor.backend_problem is None, (factory, editor.backend_problem)
added = entities()
for path in ('world/contact_sensor', 'world/frame_transformer', 'world/height_scanner_2', 'world/camera'):
    assert added[path]['mount'] == {{'entity': 'world/robot', 'body': 'base'}}, added[path]
assert added['world/frame_transformer']['parameters']['target_body'] == 'base'
assert added['world/height_scanner_2']['meshes'] == ['world/ground']
editor.select(('entity', 'world/frame_transformer'))  # The reported frame is picked from the mount's links.
assert editor.parameter_fields['target_body'].options == ('base', 'FL_foot', 'FR_foot')
editor._delete('world/ground')
assert '将没有射线目标' in next(text for text in texts if text.startswith('删除 **ground**'))
editor._remove('world/ground', editor.deletion_plan('world/ground'))
assert editor.backend_problem and entities()['world/height_scanner']['meshes'] == []
editor._choose_asset('ground_plane')  # A new ground becomes the target of scanners aiming at all ground.
editor._add_asset()
after = entities()
assert after['world/height_scanner']['meshes'] == after['world/height_scanner_2']['meshes'] == ['world/ground_plane']
assert editor.backend_problem is None, editor.backend_problem
editor._choose_asset('ground_plane')
editor._add_asset()
assert entities()['world/height_scanner']['meshes'] == ['world/ground_plane', 'world/ground_plane_2']
assert editor.backend_problem is None, editor.backend_problem
editor._remove('world/robot', editor.deletion_plan('world/robot'))  # isaaclab3: only a camera stays, fixed in the world
left = entities()
assert {{path for path, item in left.items() if item['category'] == 'sensor'}} == {{'world/camera'}}, sorted(left)
assert not left['world/camera'].get('mount'), left['world/camera']
assert editor.backend_problem is None, editor.backend_problem
editor._choose_asset('box')
editor._add_asset()
editor.items['world/box']['exported'] = True  # Exported without a body: a static box (no mass) holds no sensor.
editor.select(('entity', 'world/box'))
editor._choose_asset('camera')
assert 'world/box' not in editor.mount_targets.values(), editor.mount_targets
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
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


@pytest.mark.skipif(find_spec('viser') is None, reason='viser (web editor) not installed')
def test_new_environment_names_its_backend_and_keeps_observation_order(tmp_path):
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import yaml
import tools.web_editor as W
from shared.assembly import starter
starter.PROJECTS = __import__('pathlib').Path({str(tmp_path)!r})
editor = W.Editor({str(ROOT / 'projects/planner_executor/scene.yaml')!r}, port=0, native_preview=False)
editor.new_name.value, editor.new_template.value = 'fresh_nav', 'planner_executor'
editor._new_environment()
assert editor.draft == 'fresh_nav' and editor.snapshot.get('control'), editor.status.content
# Untick an observation, save, tick it again: it returns to its old slot.
role = editor.train_role
before = [key for key in editor.observation_order[role] if editor.observation_enabled.get((role, key))]
middle = before[1]
editor.observation_enabled[(role, middle)] = False
editor.save()
assert editor.draft is None and editor.target == starter.PROJECTS / 'fresh_nav/scene.yaml'
assert editor.scene.parent == editor.target.parent and editor.scene.read_text() == editor.target.read_text()
assert [p.name for p in editor.target.parent.glob('.scene.yaml.*.editing')] == [editor.scene.name]  # only the live copy
assert yaml.safe_load((editor.scene.parent / 'train.yaml').read_text())['env']['train']['init_params']['backend'] == 'isaaclab3'
assert editor.backend_problem is None, editor.backend_problem
editor.observation_enabled[(role, middle)] = True
editor.save()
text = editor.target.read_text()
assert not text.lstrip().startswith('{{')
# scene.yaml lists a layer's inputs; another layer's result is selected by its name (planner.goal -> planner).
from shared.assembly.layers import layer_tokens
layers = yaml.safe_load(text)['policy']['layers']
assert layers[role]['inputs'] == layer_tokens(before, layers)
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


@pytest.mark.skipif(find_spec('viser') is None or find_spec('trimesh') is None, reason='viser or trimesh not installed')
def test_textured_part_is_one_glb_with_a_primitive_per_surface(tmp_path):
    import struct
    from PIL import Image
    import tools.web_editor as W
    Image.new('RGB', (512, 512), (200, 120, 40)).save(tmp_path / 'wood.png')
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], dtype=np.uint32)
    surface = {'uv': np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [0, 1]], dtype=np.float32),
               'pieces': [{'faces': 2, 'texture': str(tmp_path / 'wood.png'), 'tint': [1, 1, 1], 'color': [.8, .5, .2]},
                          {'faces': 2, 'texture': None, 'tint': None, 'color': [.1, .1, .1]}]}
    data = W.surface_glb(vertices, faces, surface, (178, 189, 201))
    length = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20 + length])
    assert len(gltf['meshes']) == 2 and len(gltf.get('images', [])) == 1
    assert W.surface_glb(vertices, faces, surface, (178, 189, 201)) is data  # cached per arrays
    assert max(W._texture_image(str(tmp_path / 'wood.png')).size) == W.TEXTURE_PIXELS


def test_editor_app_reads_and_edits_the_selection(tmp_path):
    """The app's panels (tools/editor_ui) go through Editor.ui_get / ui_post: the same edits and undo as the view."""
    project = tmp_path / 'navigation'
    shutil.copytree(ROOT / 'projects/planner_executor', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    scene = project / 'scene.yaml'
    scene.write_text(scene.read_text().replace('\npolicy:', '\n- name: cube\n  factory: dex_cube\npolicy:', 1))
    script = f'''
import sys
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
from shared.assembly.editing import SceneDocument
editor = W.Editor({str(project / 'scene.yaml')!r}, port=0, native_preview=False)
# Nothing selected: the scene's policy (components, robots) fills the panel.
overview = editor.ui_get('inspector', {{}})
assert overview['owner'] is None and not editor.ui_get('state', {{}})['dirty']
assert [c['role'] for c in overview['scene']['components']] == ['planner', 'executor'], overview['scene']
assert [r['key'] for r in overview['scene']['robots']] == ['robot'] and overview['scene']['automatic']
assert set(editor.ui_get('state', {{}})['simulation']) >= {{'physics', 'renderer'}}
editor.ui_post('gizmo', {{'mode': 'scale'}})
assert editor.ui_get('state', {{}})['gizmo'] == 'scale' and editor.gizmo.disable_rotations and editor.gizmo.disable_sliders
editor.ui_post('gizmo', {{'mode': 'move'}})
# Scale is part of the transform, not of the raw parameters.
editor.outliner_post('select', {{'id': 'entity:world/cube'}})
data = editor.ui_get('inspector', {{}})
assert data['scale'] == [0.8, 0.8, 0.8] and 'scale' not in [f['name'] for f in data.get('parameters', [])]  # the asset's default
editor.ui_post('scale', {{'value': [2.0, 2.0, 1.0]}})
assert editor.ui_get('inspector', {{}})['scale'] == [2.0, 2.0, 1.0]
editor.ui_post('undo', {{}})
assert editor.ui_get('inspector', {{}})['scale'] == [0.8, 0.8, 0.8]
editor.outliner_post('select', {{'id': 'entity:world/destination'}})
data = editor.ui_get('inspector', {{}})
assert data['shape']['shape'] == 'point' and data['reset']['region'] == 'world/arena' and 'world/arena' in data['reset']['areas']
editor.ui_post('pose', {{'position': [1.0, 2.0, 0.0]}})
assert editor.ui_get('inspector', {{}})['position'][:2] == [1.0, 2.0] and editor.ui_get('state', {{}})['dirty']
editor.ui_post('reset', {{'region': None}})
assert editor.ui_get('inspector', {{}})['reset']['region'] is None
editor.ui_post('undo', {{}})
editor.ui_post('undo', {{}})
assert editor.ui_get('inspector', {{}})['reset']['region'] == 'world/arena' and not editor.ui_get('state', {{}})['dirty']
editor.ui_post('attach', {{'target': 'world/robot'}})  # an abstract entity follows the robot's root
parent = editor.ui_get('inspector', {{}})['parent']
assert (parent['kind'], parent['entity'], parent['mode']) == ('follow', 'world/robot', 'physical') and parent['modes'][0] == ['physical', '跟随']
editor.ui_post('parent', {{'target': None}})
assert 'parent' not in editor.ui_get('inspector', {{}})
editor.outliner_post('select', {{'id': 'entity:world/robot'}})
data = editor.ui_get('inspector', {{}})
assert data['mode'] == 'joint_position' and [field['name'] for field in data['parameters']] == ['actuators']
editor.ui_post('parameter', {{'name': 'actuators', 'value': 'supermimic'}})
editor.ui_post('label', {{'value': '机器狗'}})  # not an identifier: only the name shown
editor.ui_post('save', {{}})
saved = {{item['path']: item for item in SceneDocument(editor.scene).snapshot()['entities']}}
assert saved['world/robot']['parameters'].get('actuators') == 'supermimic' and saved['world/robot']['label'] == '机器狗'
# An identifier renames the entity and what refers to it (observations, mounts), not its factory.
before = {{key for keys in editor.observation_order.values() for key in keys if key.startswith('robot.')}}
editor.ui_post('label', {{'value': 'dog'}})
document = SceneDocument(editor.scene)
saved = {{item['path']: item for item in document.snapshot()['entities']}}
assert 'world/robot' not in saved and saved['world/dog']['factory'] == 'go2' and editor.selected == ('entity', 'world/dog')
assert saved['world/dog']['label'] == 'dog'
observed = [key for layer in document.data['policy']['layers'].values() for key in layer['inputs']]
assert observed and not any(key.startswith('robot.') for key in observed) and any(key.startswith('dog.') for key in observed), observed
assert all(item['mount']['entity'] == 'world/dog' for item in saved.values() if (item.get('mount') or {{}}).get('entity', '').endswith('dog'))
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]


def test_editor_app_groups_parents_cameras_and_imports(tmp_path):
    """Layout groups (Ctrl+G), the parent relation and link, the camera view and file imports through the app."""
    project = tmp_path / 'navigation'
    shutil.copytree(ROOT / 'projects/planner_executor', project, ignore=shutil.ignore_patterns('.cache', '__pycache__'))
    scene = project / 'scene.yaml'
    scene.write_text(scene.read_text().replace('\npolicy:', '\n- name: cube\n  factory: dex_cube\n- name: cam\n  factory: camera\n'
                                               '  pose: {position: [1, 0, 1], orientation: [1, 0, 0, 0]}\npolicy:', 1))
    import trimesh
    trimesh.creation.box((0.2, 0.4, 0.1)).export(tmp_path / 'crate.glb')
    script = f'''
import sys, time
sys.path.insert(0, {str(ROOT)!r})
import tools.web_editor as W
import shared.asset_import as A
from pathlib import Path
W.LIBRARY, A.IMPORTED = Path({str(tmp_path / 'library.json')!r}), Path({str(tmp_path / 'imported')!r})
editor = W.Editor({str(scene)!r}, port=0, native_preview=False)
# The app hides viser's panels: they are not built, and errors go to the app's toast only.
editor.ui_post('classic', {{'on': False}})
editor.outliner_post('select', {{'id': 'entity:world/cube'}})
assert not editor.inspector_items
# Parent: physical (fixed) or only laid out, chosen in place.
editor.ui_post('attach', {{'target': 'world/robot', 'mode': 'layout'}})
parent = editor.ui_get('inspector', {{}})['parent']
assert parent['mode'] == 'layout' and [m for m, _ in parent['modes']] == ['physical', 'layout'] and parent['body'] == ''
editor.ui_post('undo', {{}})
assert 'parent' not in editor.ui_get('inspector', {{}})
# The parent dialog: what the target offers, then the cube moved onto the robot's frame.
options = editor.ui_get('attach', {{'target': 'world/robot'}})
assert options['mode'] == 'physical' and [m for m, _ in options['modes']] == ['physical', 'layout'] and options['linkable']
editor.ui_post('attach', {{'target': 'world/robot', 'mode': 'layout', 'body': '', 'place': 'align'}})
assert editor.ui_get('inspector', {{}})['position'] == [0.0, 0.0, 0.0]
editor.ui_post('undo', {{}})
# Ctrl+G: the selection becomes one layout group; ungrouping keeps the places.
editor.outliner_post('select', {{'id': 'entity:world/cam', 'mode': 'toggle'}})
editor.ui_post('group', {{}})
assert editor.selected == ('group', 'group_1') and set(editor.snapshot['layout_groups']['group_1']['entities']) == {{'world/cube', 'world/cam'}}
editor.ui_post('ungroup', {{}})
assert not editor.snapshot.get('layout_groups')
# A camera shows its view (rendered by a browser; none is connected here).
editor.outliner_post('select', {{'id': 'entity:world/cam'}})
data = editor.ui_get('inspector', {{}})
assert data['camera']['width'] == 160 and [p['name'] for p in data['parameters']][:2] == ['width', 'height']
assert editor.ui_camera() is None
# The policy page names what each component reads, and the camera's images are typed choices.
scene_page = editor.ui_scene()
inputs = {{component['role']: component['inputs'] for component in scene_page['components']}}
assert all(group['rows'] and (group['path'] is None or group['icon']) for groups in inputs.values() for group in groups), inputs
assert not any(row['key'] == 'simulation.gravity' and not row['enabled']
               for groups in inputs.values() for group in groups for row in group['rows'])  # a constant: not offered
# A whole input group dropped at once is one undo step.
role, group = next((role, group) for role, groups in inputs.items() for group in groups if group['read'] > 1)
keys = [row['key'] for row in group['rows'] if row['enabled']]
editor.ui_post('unread', {{'role': role, 'keys': keys}})
assert not any(editor.observation_enabled.get((role, key)) for key in keys)
editor.ui_post('undo', {{}})
assert all(editor.observation_enabled.get((role, key)) for key in keys)
editor.ui_post('parameter', {{'name': 'data_types', 'value': ['rgb', 'distance_to_image_plane']}})
keys = [q['key'] for q in editor.ui_get('inspector', {{}})['signals']]
assert any('distance_to_image_plane' in key or 'depth' in key for key in keys), keys  # the catalog follows at once
# A sensor from the context menu: mounted on the robot's root link (its links come from the native export,
# reused from the shared mesh cache when there is one) and selected; without links it says so.
try:
    editor.ui_post('add_sensor', {{'factory': 'camera', 'target': 'world/robot'}})
    added = editor.items[editor.selected[1]]
    assert added['factory'] == 'camera' and added['mount']['entity'] == 'world/robot' and added['mount']['body'], added
except ValueError as error:
    assert '连杆名尚未解析' in str(error), error
# Selecting a contact sensor or a height scanner draws what it measures.
editor.outliner_post('select', {{'id': 'entity:world/contact_forces'}})
assert editor.sensor_marker and editor.ui_get('inspector', {{}})['parameters'][0]['name'] == 'update_period'
editor.outliner_post('select', {{'id': 'entity:world/height_scanner'}})
assert editor.sensor_marker
editor.outliner_post('select', {{'id': 'entity:world/cube'}})
assert not editor.sensor_marker
# Import: a mesh file uploaded, converted to USD in the background, listed on the shelf and added standing on the ground.
with open({str(tmp_path / 'crate.glb')!r}, 'rb') as stream:
    editor.import_file('abcdefgh12', 'crate.glb', stream, Path({str(tmp_path / 'crate.glb')!r}).stat().st_size)
assert editor.import_finish({{'batch': 'abcdefgh12', 'kind': 'object', 'name': 'crate'}})['name'] == 'crate'
for _ in range(300):
    if not any(job['running'] for job in editor.jobs):
        break
    time.sleep(.1)
assert not any(job['error'] for job in editor.jobs), editor.jobs
item = next(item for item in editor.snapshot['entities'] if item['name'] == 'crate')
# A user's import is the project's: in its assets/, on its own shelf, not the workspace's.
own = Path(editor.target).parent / 'assets'
assert item['factory'] == 'usd_object' and item['parameters']['usd_path'] == str(own / 'crate/usd/crate.usda')
assert abs(item['pose']['position'][2] - 0.2) < 1e-6  # glTF is Y-up: its 0.4 m Y side stands up, the bottom on the ground
assert [entry['id'] for entry in W.load_library(own / 'library.json')] == ['crate'] and not W.load_library()
assert 'project:crate' in editor.gallery_entries and not any(A.IMPORTED.glob('*'))
editor.server.stop()
'''
    result = subprocess.run([str(EDITOR_PYTHON), '-c', script], cwd=ROOT, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

pytestmark = pytest.mark.group_editor  # runs in the [editor] dependency group
