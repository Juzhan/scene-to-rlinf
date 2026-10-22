"""Source preservation and real Blender round trips for unified scene authoring."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from ruamel.yaml import YAML

from shared.assembly.editing import EditConflict, SceneDocument, apply_edits

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path):
    destination = tmp_path / "project"
    shutil.copytree(ROOT / "projects/assembly_lift", destination,
                    ignore=shutil.ignore_patterns("__pycache__", "__init__.py"))
    return destination / "scene.yaml"


def load(path):
    return YAML(typ="safe").load(path.read_text())


def test_python_task_survives_scene_edits(project):
    task = (project.parent / 'task.py').read_bytes()
    document = SceneDocument(project)
    document.set_entity('world/object', {'pose': {'position': [.6, 0, .055]}})
    assert document.save()
    assert (project.parent / 'task.py').read_bytes() == task
    assert 'task' not in load(project)
    assert document.snapshot()['task_source'].endswith('task.py')

def test_noop_keeps_original_bytes(project):
    original = project.read_bytes()
    document = SceneDocument(project)
    assert not document.save()
    assert project.read_bytes() == original
    assert not apply_edits(project, {"revision": document.revision})["changed"]
    assert project.read_bytes() == original


def test_display_labels_do_not_break_entity_references(project):
    document = SceneDocument(project)
    result = apply_edits(project, {'revision': document.revision, 'labels': {'world/robot': '主机械臂'}})
    robot = next(item for item in result['snapshot']['entities'] if item['path'] == 'world/robot')
    assert robot['label'] == '主机械臂'
    assert robot['name'] == 'robot'
    assert result['snapshot']['control']['output'] == document.data['control']['output']


def test_unfinished_environment_can_be_authored_but_not_trained(tmp_path):
    from shared.backends.isaac.task_config import validate_task_settings
    path = tmp_path / 'scene.yaml'
    path.write_text('schema_version: 1\nname: world\nentities: []\n')
    result = apply_edits(path, {'revision': SceneDocument(path).revision,
                              'add_entities': [{'name': 'table', 'factory': 'seattle_table'}]})
    assert result['snapshot']['entities'][0]['name'] == 'table'
    with pytest.raises(ValueError, match='observations'):
        validate_task_settings(load(path))


def test_python_task_rejects_formula_payload_without_writing(project):
    document = SceneDocument(project)
    before = project.read_bytes()
    with pytest.raises(ValueError, match='task.py'):
        apply_edits(project, {'revision': document.revision, 'variables': {'x': {'expr':'1'}}})
    assert project.read_bytes() == before

def test_template_child_changes_only_instance(project):
    template = project.parent / "assemblies/arm.yaml"
    template_bytes = template.read_bytes()
    document = SceneDocument(project)
    # A second instance shares the same source but must keep its original pose.
    document.data["entities"].append({"name": "second_robot", "assembly": "assemblies/arm.yaml"})
    document._index()
    document.set_entity("world/robot/wrist_camera", {"pose": {"position": [.3, .1, .2]}})
    document.save()
    settings = load(project)
    instance = next(item for item in settings["entities"] if item["name"] == "robot")
    second = next(item for item in settings["entities"] if item["name"] == "second_robot")
    assert instance["assembly"] == "assemblies/arm.yaml"
    assert instance["overrides"]["wrist_camera"]["pose"]["position"] == [.3, .1, .2]
    assert "children" not in instance
    assert "overrides" not in second
    assert template.read_bytes() == template_bytes
    reloaded = SceneDocument(project)
    assert reloaded.entities["world/second_robot/wrist_camera"]["pose"]["position"] == [.2, 0, .06]
    assert reloaded.entities["world/robot/wrist_camera"]["mount"]["body"] == "panda_hand"


def test_root_pose_uses_instance_pose_and_keeps_omitted_defaults(project):
    document = SceneDocument(project)
    document.set_entity("world/robot", {"pose": {"position": [.1, .2, 0]}})
    document.save()
    raw = load(project)
    robot = next(item for item in raw["entities"] if item["name"] == "robot")
    assert robot["pose"]["position"] == [.1, .2, 0]
    assert "overrides" not in robot
    assert "pose" not in next(item for item in raw["entities"] if item["name"] == "light")


def test_unknown_fields_comments_and_aliases_survive(project):
    (project.parent / 'task.py').unlink()
    project.write_text(project.read_text() + '\n# custom integration metadata\neditor_note: {keep_me: 7}\n')
    document = SceneDocument(project)
    document.set_entity("world/object", {"pose": {"position": [.6, 0, .055]}})
    document.save()
    assert load(project)["editor_note"] == {"keep_me": 7}
    assert "# custom integration metadata" in project.read_text()


def test_alias_parameter_edits_do_not_change_other_entities(tmp_path):
    path = tmp_path / "scene.yaml"
    path.write_text('''schema_version: 1
name: world
entities:
  - name: a
    factory: camera
    parameters: &params {width: 160, height: 120}
  - name: b
    factory: camera
    parameters: *params
''')
    document = SceneDocument(path)
    document.set_entity("world/a", {"parameters": {"width": 320}})
    document.save()
    entities = load(path)["entities"]
    assert entities[0]["parameters"]["width"] == 320
    assert entities[1]["parameters"]["width"] == 160


@pytest.mark.parametrize("reference", [False, True])
def test_external_changes_refuse_write(project, reference):
    document = SceneDocument(project)
    document.set_entity("world/object", {"pose": {"position": [.7, 0, .055]}})
    changed = project.parent / "assemblies/arm.yaml" if reference else project
    changed.write_text(changed.read_text() + "\n# changed elsewhere\n")
    original = project.read_bytes()
    with pytest.raises(EditConflict, match="Changed outside"):
        document.save()
    assert project.read_bytes() == original


def test_reference_change_during_edit_is_not_rebaselined(project):
    document = SceneDocument(project)
    template = project.parent / "assemblies/arm.yaml"
    template.write_text(template.read_text() + "\n# external update\n")
    with pytest.raises(EditConflict):
        document.set_entity("world/object", {"pose": {"position": [.7, 0, .055]}})


def test_world_frame_edits_rejected(project):
    document = SceneDocument(project)
    with pytest.raises(ValueError, match="body-local"):
        document.set_entity("world/robot/wrist_camera", {"pose": {"position": [1, 2, 3]}}, frame="world")


def test_object_observation_selection_is_a_plain_field_list(project):
    document = SceneDocument(project)
    document.set_observations('states', ['robot.joint_position','object.position'])
    document.save()
    assert load(project)['control']['components']['executor']['observations'] == ['robot.joint_position','object.position']
    assert 'class ExecutorObs' in (project.parent / 'observation.py').read_text()

def test_region_and_reset_written_in_same_validated_edit(project):
    document = SceneDocument(project)
    result = apply_edits(project, {"revision": document.revision,
        "add_entities": [{"name": "cube_spawn", "kind": "abstract", "shape": "rectangle", "size": [.3, .4],
                          "pose": {"position": [.5, 0, .055]}}],
        "entities": [{"path": "world/object", "changes": {"reset": {"region": "world/cube_spawn", "yaw_range": [-.3, .3]}}}]})
    assert result["changed"]
    settings = load(project)
    spawn = next(item for item in settings["entities"] if item["name"] == "cube_spawn")
    assert spawn["kind"] == "abstract" and spawn["size"] == [.3, .4]
    assert next(item for item in settings["entities"] if item["name"] == "object")["reset"]["region"] == "world/cube_spawn"
    assert "regions" not in settings


def test_abstract_entity_reset_and_display_are_entity_edits(tmp_path):
    from shared.assembly.abstract import abstract_catalog
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities: []\n')
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'add_entities': [
        {'name': 'arena', 'kind': 'abstract', 'shape': 'rectangle', 'size': [6., 6.], 'pose': {'position': [0., 0., .4]}},
        {'name': 'destination', 'kind': 'abstract', 'shape': 'point'}]})
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'visualization': {'abstract': ['destination']},
                        'entities': [{'path': 'world/destination', 'changes': {
                            'reset': {'region': 'world/arena', 'yaw_range': [-1., 1.], 'interval': 4.}}}]})
    saved = load(scene)
    destination = next(item for item in saved['entities'] if item['name'] == 'destination')
    assert destination['reset'] == {'region': 'world/arena', 'yaw_range': [-1., 1.], 'interval': 4.}
    assert saved['visualization'] == {'abstract': ['destination']}
    assert set(abstract_catalog(saved)) == {'arena.position', 'arena.orientation', 'arena.size',
                                            'destination.position', 'destination.orientation'}
    # Resizing the area keeps the sampled entity; null makes it static again.
    apply_edits(scene, {'revision': SceneDocument(scene).revision,
                        'entities': [{'path': 'world/arena', 'changes': {'size': [4., 5.]}},
                                     {'path': 'world/destination', 'changes': {'reset': None}}]})
    saved = load(scene)
    assert next(item for item in saved['entities'] if item['name'] == 'arena')['size'] == [4., 5.]
    assert 'reset' not in next(item for item in saved['entities'] if item['name'] == 'destination')
    with pytest.raises(ValueError, match='Only abstract entities'):
        apply_edits(scene, {'revision': SceneDocument(scene).revision, 'visualization': {'abstract': ['missing']}})


def test_positions_and_spaces_are_deleted_with_the_scene_edit(tmp_path):
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities: []\n')
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'add_entities': [
        {'name': 'drop', 'kind': 'abstract', 'shape': 'point', 'pose': {'position': [1., 2., 3.]}},
        {'name': 'keep', 'kind': 'abstract', 'shape': 'rectangle', 'size': [1., 1.]},
        {'name': 'box', 'factory': 'dex_cube'}]})
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'visualization': {'abstract': ['drop', 'keep']},
                        'entities': [{'path': 'world/box', 'changes': {'reset': {'region': 'world/keep'}}}]})
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'remove_entities': ['world/drop']})
    saved = load(scene)
    assert [item['name'] for item in saved['entities']] == ['keep', 'box']
    assert saved['visualization'] == {'abstract': ['keep']}
    with pytest.raises(ValueError, match='Unknown entity'):
        apply_edits(scene, {'revision': SceneDocument(scene).revision, 'remove_entities': ['world/drop']})
    # A referenced area cannot be deleted silently.
    before = scene.read_bytes()
    with pytest.raises(ValueError, match='先修改其复位方式'):
        apply_edits(scene, {'revision': SceneDocument(scene).revision, 'remove_entities': ['world/keep']})
    assert scene.read_bytes() == before


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed")
def test_blender_deletes_positions_and_spaces_with_the_scene(tmp_path):
    script = tmp_path / "delete_abstract.py"
    script.write_text('''
import sys
sys.path.insert(0, ROOT_PATH)
import bpy
from tools.blender_addon import register, ui
register()
ui._SESSION.clear()
item = {"path": "world/draft_view", "name": "draft_view", "category": "abstract", "shape": "rectangle",
        "size": [1., 1.], "pose": {"position": [0., 0., 0.], "orientation": [1., 0., 0., 0.]}}
ui._SESSION["snapshot"] = {"revision": "test", "entities": [item], "catalog": {"observations": []}}
obj = ui._abstract(item)
obj["rl_entity"] = "world/draft_view"
assert not ui.collect_edits(bpy.context).get("remove_entities")
bpy.data.objects.remove(obj)
assert ui.collect_edits(bpy.context)["remove_entities"] == ["world/draft_view"]
print("ABSTRACT_DELETE_OK")
'''.replace("ROOT_PATH", repr(str(ROOT))))
    result = subprocess.run([shutil.which("blender"), "--background", "--factory-startup",
        "--python-exit-code", "1", "--python", str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ABSTRACT_DELETE_OK" in result.stdout


def test_invalid_edit_does_not_write(project):
    document = SceneDocument(project)
    original = project.read_bytes()
    with pytest.raises(ValueError, match="factory"):
        apply_edits(project, {"revision": document.revision, "entities": [
            {"path": "world/object", "changes": {"parameters": {"not_a_cube_parameter": 5}}}]})
    assert project.read_bytes() == original
    with pytest.raises(ValueError, match="positive"):
        apply_edits(project, {"revision": document.revision, "add_entities": [
            {"name": "broken", "kind": "abstract", "shape": "box", "size": [1, -1, 1]}]})
    assert project.read_bytes() == original


def test_action_type_switch_drops_incompatible_old_parameters(project):
    document = SceneDocument(project)
    joints = [f"panda_joint{i}" for i in range(1, 8)]
    with pytest.raises(ValueError, match="逐个列出"):
        document.set_action("arm", {"type": "joint_effort", "asset_name": "robot", "joint_names": ["panda_joint.*"]})
    document.set_action("arm", {"type": "joint_effort", "asset_name": "robot", "joint_names": joints})
    document.save()
    assert "use_default_offset" not in load(project)["control"]["output"]["arm"]


def test_task_settings_are_not_written_into_scene(project):
    before = project.read_bytes()
    with pytest.raises(ValueError, match='task.py'):
        apply_edits(project, {'revision': SceneDocument(project).revision, 'task': {'reward':'1'}})
    assert project.read_bytes() == before

def test_invalid_reward_keeps_source(project):
    document = SceneDocument(project)
    original = project.read_bytes()
    with pytest.raises(ValueError, match="finite"):
        apply_edits(project, {"revision": document.revision, "rewards": {"approach_object": {"weight": float("nan")}}})
    assert project.read_bytes() == original


def test_apply_requires_revision(project):
    with pytest.raises(ValueError, match="revision"):
        apply_edits(project, {"entities": []})


@pytest.mark.skipif(shutil.which("blender") is None, reason="Blender not installed")
def test_real_blender_edits_save_and_reload(project, tmp_path):
    script = tmp_path / "roundtrip.py"
    report = tmp_path / "report.json"
    script.write_text('''import sys, json
from pathlib import Path
import bpy
sys.path.insert(0, ROOT)
from tools.blender_addon import register, ui
register()
bpy.context.scene.rl_project_scene = SCENE
bpy.context.scene.rl_project_python = PYTHON
ui.load_project(bpy.context)
ui.verify_round_trip(bpy.context)
objects = ui._entity_objects(bpy.context)
objects['world/object'].location.x += .1
objects['world/robot/wrist_camera'].location.y += .05
objects['world/spawn_area'].scale.x = .3
ui._SESSION['pending_resets']['world/object'] = {'region':'world/spawn_area', 'yaw_range':[-.2,.2]}
for item in bpy.context.scene.rl_observations:
    if item.profile == 'executor' and item.term_id == 'robot.joint_velocity': item.enabled = False
assert not hasattr(bpy.context.scene, 'rl_rewards')
assert not hasattr(ui, 'RLPROJECT_PT_rewards')
saved = ui.save_project(bpy.context)
assert saved['changed']
ui.load_project(bpy.context)
assert not ui.save_project(bpy.context)['changed']
Path(REPORT).write_text(json.dumps({'changed':True,'no_second_write':True}))
'''.replace("ROOT", repr(str(ROOT))).replace("SCENE", repr(str(project))).replace("PYTHON", repr(sys.executable)).replace("REPORT", repr(str(report))))
    completed = subprocess.run([shutil.which("blender"), "--background", "--factory-startup", "--python-exit-code", "1", "--python", str(script)],
                               capture_output=True, text=True, timeout=90)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(report.read_text())["no_second_write"]
    settings = load(project)
    object_cfg = next(item for item in settings["entities"] if item["name"] == "object")
    robot_cfg = next(item for item in settings["entities"] if item["name"] == "robot")
    assert object_cfg["pose"]["position"][0] == pytest.approx(.6)
    assert object_cfg["reset"] == {"region": "world/spawn_area", "yaw_range": [-.2, .2]}
    assert robot_cfg["overrides"]["wrist_camera"]["pose"]["position"][1] == pytest.approx(.05)
    assert "robot.joint_velocity" not in settings["control"]["components"]["executor"]["observations"]
    spawn_cfg = next(item for item in settings["entities"] if item["name"] == "spawn_area")
    assert spawn_cfg["size"] == pytest.approx([.3, .5])
    assert "regions" not in settings
    assert "task" not in settings


def test_camera_mount_and_observation_channels_preserve_templates(project):
    document = SceneDocument(project)
    path = 'world/robot/wrist_camera'
    source = (project.parent / 'assemblies/arm.yaml').read_bytes()
    result = apply_edits(project, {'revision': document.revision, 'entities': [
        {'path': path, 'changes': {'mount': {'entity': 'world/robot', 'body': 'panda_link0'},
                                  'pose': {'position': [.1, 0, .3]}}}],
        'observations': {'executor': ['robot.wrist_camera.rgb']}})
    assert result['changed']
    settings = load(project)
    assert settings['application']['enable_cameras']
    assert settings['control']['components']['executor']['observations'] == ['robot.wrist_camera.rgb']
    assert settings['control']['components']['planner']['observations'] == ['object.target.position']
    assert (project.parent / 'assemblies/arm.yaml').read_bytes() == source
    assert SceneDocument(project).entities[path]['mount']['body'] == 'panda_link0'


def test_connection_cycle_and_layout_group_overlap_do_not_write(project):
    document = SceneDocument(project)
    before = project.read_bytes()
    with pytest.raises(ValueError, match='Cyclic'):
        apply_edits(project, {'revision': document.revision, 'connections': {
            'a': {'child': {'entity': 'world/robot'}, 'parent': {'entity': 'world/table'}},
            'b': {'child': {'entity': 'world/table'}, 'parent': {'entity': 'world/robot'}}}})
    assert project.read_bytes() == before
    document.set_layout_group('bench', {'entities': ['world/table', 'world/robot']})
    document.save()
    before = project.read_bytes()
    with pytest.raises(ValueError, match='another layout group'):
        apply_edits(project, {'revision': SceneDocument(project).revision,
                             'layout_groups': {'other': {'entities': ['world/robot']}}})
    assert project.read_bytes() == before


@pytest.mark.skipif(shutil.which('blender') is None, reason='Blender not installed')
def test_blender_layout_group_moves_saved_entities_without_renaming(project, tmp_path):
    script = tmp_path / 'group.py'
    script.write_text('''import sys
import bpy
sys.path.insert(0, ROOT)
from tools.blender_addon import register, ui
register()
bpy.context.scene.rl_project_scene=SCENE
bpy.context.scene.rl_project_python=PYTHON
ui.load_project(bpy.context)
objects=ui._entity_objects(bpy.context)
bpy.ops.object.select_all(action='DESELECT')
for path in ('world/table','world/robot'): objects[path].select_set(True)
bpy.context.view_layer.objects.active=objects['world/robot']
assert bpy.ops.rl_project.layout_group(name='bench') == {'FINISHED'}
original={key:obj.matrix_world.translation.copy() for key,obj in ui._entity_objects(bpy.context).items()}
group=bpy.context.object
group.location.x += .7
assert ui.save_project(bpy.context)['changed']
ui.load_project(bpy.context)
for key in ('world/table','world/robot'):
    assert abs(ui._entity_objects(bpy.context)[key].matrix_world.translation.x-original[key].x-.7)<1e-5
assert not ui.save_project(bpy.context)['changed']
# Filtering never disables hidden observations, and new candidates stay unchecked.
assert any(item.term_id.startswith('robot.') and not item.enabled for item in bpy.context.scene.rl_observations)
assert [item.term_id for item in bpy.context.scene.rl_observations if item.enabled and item.profile=='executor']==['robot.joint_position','robot.joint_velocity','object.position','planner.output','ee_frame.frame_positions','robot.default_joint_position']
group=next(obj for obj in bpy.context.scene.objects if obj.get('rl_layout_group')=='bench')
bpy.context.view_layer.objects.active=group
assert bpy.ops.rl_project.ungroup()=={'FINISHED'}
assert abs(ui._entity_objects(bpy.context)['world/robot'].matrix_world.translation.x-original['world/robot'].x-.7)<1e-5
'''.replace('ROOT',repr(str(ROOT))).replace('SCENE',repr(str(project))).replace('PYTHON',repr(sys.executable)))
    result=subprocess.run([shutil.which('blender'),'--background','--factory-startup','--python-exit-code','1','--python',str(script)],capture_output=True,text=True,timeout=90)
    assert result.returncode==0, result.stdout+result.stderr
    settings=load(project)
    assert settings['control']['output']['arm']['asset_name']=='robot'
    assert not settings['authoring']['groups']


def test_delete_camera_then_add_piper(project):
    doc = SceneDocument(project)
    doc.add_entity('camera', 'camera', kind='sensor')
    doc.set_observations('executor', ['camera.rgb', 'robot.joint_position'])
    doc.save()
    result = apply_edits(project, {'revision': SceneDocument(project).revision,
                                  'remove_entities': ['world/camera'],
                                  'add_entities': [{'name': 'piper', 'factory': 'piper', 'kind': 'robot'}]})
    assert 'world/camera' not in SceneDocument(project).entities
    assert 'world/piper' in SceneDocument(project).entities
    assert 'camera.rgb' not in result['snapshot']['observations']['executor']


def test_layout_parent_cycle_rejected_atomically(project):
    original = project.read_bytes()
    with pytest.raises(ValueError, match='循环'):
        apply_edits(project, {'revision': SceneDocument(project).revision,
                             'layout_parents': {'world/robot': {'entity': 'world/table'},
                                                'world/table': {'entity': 'world/robot'}}})
    assert project.read_bytes() == original


def test_document_delete_requires_detaching_mounted_sensors(project):
    # The editor detaches or drops them in the same edit, since only it knows their live world pose.
    original = project.read_bytes()
    with pytest.raises(ValueError, match='挂载在被删除的实体上'):
        apply_edits(project, {'revision': SceneDocument(project).revision, 'remove_entities': ['world/robot']})
    assert project.read_bytes() == original


def test_deleting_a_parent_keeps_children_in_place(tmp_path):
    scene = tmp_path / 'scene.yaml'
    scene.write_text("""schema_version: 1
name: world
entities:
  - name: stand
    kind: scene
    pose: {position: [1.0, 0.0, 0.0], orientation: [0.7071068, 0.0, 0.0, 0.7071068]}
    children:
      - name: cube  # kept comment
        factory: dex_cube
        pose: {position: [0.5, 0.0, 0.0]}
  - name: tray
    factory: dex_cube
  - name: lid
    factory: dex_cube
authoring:
  parents:
    world/tray: {entity: world/stand/cube}
    world/lid: {entity: world/stand}
  labels:
    world/stand/cube: 方块
""")
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'remove_entities': ['world/stand']})
    saved = load(scene)
    assert [item['name'] for item in saved['entities']] == ['cube', 'tray', 'lid']
    assert np.allclose(saved['entities'][0]['pose']['position'], [1.0, 0.5, 0.0], atol=1e-6)
    assert saved['authoring']['parents'] == {'world/tray': {'entity': 'world/cube'}}
    assert saved['authoring']['labels'] == {'world/cube': '方块'}
    assert '# kept comment' in scene.read_text()


@pytest.mark.skipif(not shutil.which('blender'), reason='Blender unavailable')
def test_blender_delete_parenting_and_align(project, tmp_path):
    doc = SceneDocument(project)
    doc.data['entities'].append({'name': 'workstation', 'pose': {'position': [2, 0, 0]},
                                 'children': [{'name': 'lamp', 'factory': 'dome_light', 'pose': {'position': [0, 0, 1]}}]})
    doc._index()
    doc.save()
    script = tmp_path / 'hierarchy.py'
    script.write_text('''
import sys, json
sys.path.insert(0, ROOT)
import bpy
from mathutils import Vector
from tools.blender_addon import register, ui
register()
bpy.context.scene.rl_project_scene = SCENE
bpy.context.scene.rl_project_python = PYTHON
ui.load_project(bpy.context)
# Delete a root with an enabled camera observation, then use the real Add Asset operator.
payload = {'revision': ui._SESSION['snapshot']['revision'], 'add_entities': [{'name': 'camera', 'factory': 'camera', 'kind': 'sensor'}], 'observations': {'executor': ['camera.rgb', 'robot.joint_position']}}
ui.bridge(bpy.context, 'apply', payload)
ui.load_project(bpy.context)
bpy.data.objects.remove(ui._entity_objects(bpy.context)['world/camera'], do_unlink=True)
real_bridge = ui.bridge
def bridge(context, command, payload=None):
    if command == 'visuals':
        return {}  # Native geometry is covered separately; no simulator needed here.
    return real_bridge(context, command, payload)
ui.bridge = bridge
ui.start_visual_export = lambda *args, **kwargs: None
assert bpy.ops.rl_project.add_asset(name='piper', factory='piper') == {'FINISHED'}
assert 'world/camera' not in ui._entity_objects(bpy.context)
assert 'world/piper' in ui._entity_objects(bpy.context)
# Preview/cancel use real Blender transforms without touching the YAML.
from types import SimpleNamespace, MethodType
from pathlib import Path
bpy.context.view_layer.objects.active = ui._entity_objects(bpy.context)['world/piper']
original_bytes = Path(SCENE).read_bytes()
original_world = bpy.context.object.matrix_world.copy()
probe = SimpleNamespace(target='world/table', body='__root__', mode='layout', keep_layout=False,
                        offset=(.2,.1,.8), rotation=(0,0,.3), align='none')
for method in ('_begin_preview', '_restore_preview', '_offset_matrix', 'check', 'cancel'):
    setattr(probe, method, MethodType(getattr(ui.RLPROJECT_OT_attach, method), probe))
probe._begin_preview(bpy.context)
assert probe.check(bpy.context)
parent = ui._entity_objects(bpy.context)['world/table']
assert (bpy.context.object.matrix_world.translation - (parent.matrix_world @ Vector((.2,.1,.8)))).length < 1e-5
assert Path(SCENE).read_bytes() == original_bytes
probe.offset = (.1,.2,.5)
assert probe.check(bpy.context)
probe.cancel(bpy.context)
assert (bpy.context.object.matrix_world.translation - original_world.translation).length < 1e-5
assert Path(SCENE).read_bytes() == original_bytes
assert not ui._SESSION.get('preview_active')
# A child follows the selected parent's local axes; source is baked world pose.
obj = ui._entity_objects(bpy.context)['world/piper']
bpy.context.view_layer.objects.active = obj
assert bpy.ops.rl_project.attach(target='world/table', body='__root__', mode='layout', keep_layout=False, offset=(0,0,0.4), align='all') == {'FINISHED'}
objects = ui._entity_objects(bpy.context)
obj, parent = objects['world/piper'], objects['world/table']
assert obj.parent == parent
assert (obj.location - Vector((0,0,.4))).length < 1e-5
parent.location.x += .3
bpy.context.view_layer.update()
expected = obj.matrix_world.copy()
ui.save_project(bpy.context)
ui.load_project(bpy.context)
obj = ui._entity_objects(bpy.context)['world/piper']
assert (obj.matrix_world.translation - expected.translation).length < 1e-5
assert not ui.save_project(bpy.context)['changed']
bpy.context.view_layer.objects.active = obj
assert bpy.ops.rl_project.detach() == {'FINISHED'}
obj = ui._entity_objects(bpy.context)['world/piper']
assert obj.parent is None
assert (obj.matrix_world.translation - expected.translation).length < 1e-5
# Physical attachments expose the same relative placement controls, but persist
# their anchor pose rather than baking an independent child root.
assert bpy.ops.rl_project.add_asset(name='base_cube', factory='dex_cube') == {'FINISHED'}
assert bpy.ops.rl_project.add_asset(name='child_cube', factory='dex_cube') == {'FINISHED'}
bpy.context.view_layer.objects.active = ui._entity_objects(bpy.context)['world/child_cube']
assert bpy.ops.rl_project.attach(target='world/base_cube', body='__root__', mode='physical', keep_layout=False, offset=(0,0,.2)) == {'FINISHED'}
child = ui._entity_objects(bpy.context)['world/child_cube']
child.location.x = .3
ui.save_project(bpy.context)
ui.load_project(bpy.context)
child = ui._entity_objects(bpy.context)['world/child_cube']
assert (child.location - Vector((.3,0,.2))).length < 1e-5
assert not ui.save_project(bpy.context)['changed']
# All entity categories can participate in editor parenting, including a nested
# authored child whose saved pose must stay local to its original YAML parent.
assert bpy.ops.rl_project.add_asset(name='view_camera', factory='camera', category='sensor') == {'FINISHED'}
for child_path, parent_path in (
    ('world/light', 'world/floor'),
    ('world/view_camera', 'world/light'),
    ('world/piper', 'world/view_camera'),
    ('world/workstation', 'world/floor'),
    ('world/workstation/lamp', 'world/piper'),
):
    bpy.context.view_layer.objects.active = ui._entity_objects(bpy.context)[child_path]
    assert bpy.ops.rl_project.attach(target=parent_path, body='__root__', mode='layout', keep_layout=False, offset=(.1,.2,.3)) == {'FINISHED'}
    objects = ui._entity_objects(bpy.context)
    assert objects[child_path].parent == objects[parent_path]
objects['world/piper'].location.x += .25
bpy.context.view_layer.update()
expected = objects['world/workstation/lamp'].matrix_world.copy()
ui.save_project(bpy.context)
ui.load_project(bpy.context)
assert (ui._entity_objects(bpy.context)['world/workstation/lamp'].matrix_world.translation - expected.translation).length < 1e-5
assert not ui.save_project(bpy.context)['changed']
bpy.context.view_layer.objects.active = ui._entity_objects(bpy.context)['world/child_cube']
assert bpy.ops.rl_project.initial_state(linear=(.1,.2,.3), angular=(0,0,.4)) == {'FINISHED'}
updated = next(item for item in ui._SESSION['snapshot']['entities'] if item['path'] == 'world/child_cube')
assert abs(updated['initial_state']['linear_velocity'][1] - .2) < 1e-5
'''.replace('ROOT', repr(str(ROOT))).replace('SCENE', repr(str(project))).replace('PYTHON', repr(sys.executable)))
    completed = subprocess.run([shutil.which('blender'), '--background', '--factory-startup', '--python-exit-code', '1', '--python', str(script)], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_mixed_layout_physical_cycle_rejected(project):
    original = project.read_bytes()
    with pytest.raises(ValueError, match='循环'):
        apply_edits(project, {'revision': SceneDocument(project).revision,
                             'connections': {'mount': {'parent': {'entity': 'world/object'}, 'child': {'entity': 'world/table'}}},
                             'layout_parents': {'world/object': {'entity': 'world/table'}}})
    assert project.read_bytes() == original


def test_abstract_point_survives_editor_save(tmp_path):
    scene = tmp_path / 'scene.yaml'
    scene.write_text('schema_version: 1\nname: world\nentities: []\n')
    point = {'name': 'goal', 'kind': 'abstract', 'shape': 'point', 'pose': {'position': [1., 2., 3.]}}
    apply_edits(scene, {'revision': SceneDocument(scene).revision, 'add_entities': [point]})
    assert load(scene)['entities'] == [{**point, 'pose': {'position': [1., 2., 3.], 'orientation': [1., 0., 0., 0.]}}]
    item = next(item for item in SceneDocument(scene).snapshot()['entities'] if item['path'] == 'world/goal')
    assert item['category'] == 'abstract' and item['shape'] == 'point'
    assert not apply_edits(scene, {'revision': SceneDocument(scene).revision})['changed']


@pytest.mark.skipif(shutil.which('blender') is None, reason='Blender not installed')
def test_blender_component_observations_are_independent(tmp_path):
    script = tmp_path / 'roles.py'
    script.write_text("import sys\nsys.path.insert(0, ROOT_PATH)\nimport bpy\nfrom tools.blender_addon import register, ui\nregister()\nsnapshot={'revision':'test','entities':[], 'observations':{'planner':['robot.position'],'executor':['robot.joint_position']},\n 'control':{'train':'planner','components':{'planner':{'observations':['robot.position']},'executor':{'observations':['robot.joint_position']}}},\n 'catalog':{'observations':[{'id':k,'available':True} for k in ['robot.position','robot.joint_position']]},'layout_groups':{}}\nui._SESSION.clear()\nui._SESSION['snapshot']=snapshot\nui._populate_observations(bpy.context.scene,snapshot)\nassert len(bpy.context.scene.rl_observations)==4\nassert not ui.collect_edits(bpy.context)['observations']\nfor item in bpy.context.scene.rl_observations:\n if item.profile=='planner' and item.term_id=='robot.joint_position': item.enabled=True\npayload=ui.collect_edits(bpy.context)\nassert payload['observations']=={'planner':['robot.position','robot.joint_position']},payload\nassert not {'task','variables','rewards'} & payload.keys()\nassert not hasattr(ui,'RLPROJECT_PT_rewards')\nbpy.context.scene.rl_control_train = 'executor'\nassert ui.collect_edits(bpy.context)['control_train'] == 'executor'\nprint('ROLE_SELECTION_AND_PAYLOAD_OK')\n".replace('ROOT_PATH', repr(str(ROOT))))
    result = subprocess.run([shutil.which('blender'), '--background', '--factory-startup',
        '--python-exit-code', '1', '--python', str(script)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ROLE_SELECTION_AND_PAYLOAD_OK' in result.stdout


def test_delete_entity_drops_its_contact_material(project):
    doc = SceneDocument(project)
    doc.add_entity('cube_2', 'dex_cube')
    doc.data.setdefault('world', {})['contact_materials'] = {
        'cube_2': {'static_friction': .8, 'dynamic_friction': .6, 'restitution': 0.}}
    doc.save()
    apply_edits(project, {'revision': SceneDocument(project).revision, 'remove_entities': ['world/cube_2']})
    assert 'cube_2' not in (load(project).get('world') or {}).get('contact_materials', {})
